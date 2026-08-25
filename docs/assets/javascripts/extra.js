const WIND_PAIR_PATTERN = /([+\-\u2212\u00b1]?\d+(?:\.\d+)?(?:\s*[\u2013\u2014]\s*[+\-\u2212\u00b1]?\d+(?:\.\d+)?)?)\s*(m\/s\s*)?\(\s*([+\-\u2212\u00b1]?\d+(?:\.\d+)?(?:\s*[\u2013\u2014]\s*[+\-\u2212\u00b1]?\d+(?:\.\d+)?)?)\s*kt\s*\)/giu
const WIND_HEADER_PATTERN = /m\/s\s*\(kt\)/giu
const TABLE_MIN_ROWS = 2
const TABLE_SEARCH_MIN_ROWS = 8
const TABLE_PAGE_MIN_ROWS = 100
const TABLE_PAGE_SIZE = 15
const WIND_DECIMAL_PLACES = 2

function replaceTextMatches(root, pattern, replacement) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
  const textNodes = []
  let node

  while ((node = walker.nextNode())) {
    textNodes.push(node)
  }

  let replacements = 0
  textNodes.forEach((textNode) => {
    pattern.lastIndex = 0
    const text = textNode.nodeValue
    const matches = [...text.matchAll(pattern)]
    if (!matches.length) return

    const fragment = document.createDocumentFragment()
    let offset = 0
    matches.forEach((match) => {
      fragment.append(document.createTextNode(text.slice(offset, match.index)))
      fragment.append(replacement(match))
      offset = match.index + match[0].length
      replacements += 1
    })
    fragment.append(document.createTextNode(text.slice(offset)))
    textNode.replaceWith(fragment)
  })

  return replacements
}

function roundedWindValue(value) {
  return value
    .split(/(\s*[\u2013\u2014]\s*)/u)
    .map((part, index) => {
      if (index % 2) return part
      const match = part.trim().match(/^([+\-\u2212\u00b1]?)(\d+(?:\.\d+)?)$/u)
      if (!match) return part

      const rounded = Number(match[2])
        .toFixed(WIND_DECIMAL_PLACES)
        .replace(/\.?0+$/u, "")
      const sign = Number(rounded) === 0 && match[1] !== "\u00b1" ? "" : match[1]
      return `${sign}${rounded}`
    })
    .join("")
}

function unitValue(value, unit, includeUnit = true) {
  const span = document.createElement("span")
  const unitClass = unit === "m/s" ? "ms" : unit
  span.className = `wind-unit-value wind-unit-value--${unitClass}`
  span.textContent = `${roundedWindValue(value)}${includeUnit ? ` ${unit}` : ""}`
  return span
}

function prepareWindValues(table) {
  const headings = [...table.querySelectorAll("thead th")]
  const windColumns = new Set()

  headings.forEach((heading, index) => {
    WIND_HEADER_PATTERN.lastIndex = 0
    if (WIND_HEADER_PATTERN.test(heading.textContent)) windColumns.add(index)
  })

  windColumns.forEach((index) => {
    table.querySelectorAll("tbody tr").forEach((row) => {
      const cell = row.cells[index]
      if (!cell) return
      const firstNumber = cell.textContent.trim().match(/^[+\-\u2212]?\d+(?:\.\d+)?/u)
      if (firstNumber) cell.dataset.order = firstNumber[0].replace("\u2212", "-")
    })
  })

  let pairs = 0
  table.querySelectorAll("tbody td").forEach((cell) => {
    pairs += replaceTextMatches(cell, WIND_PAIR_PATTERN, (match) => {
      const fragment = document.createDocumentFragment()
      fragment.append(unitValue(match[1], "m/s", Boolean(match[2])))
      fragment.append(unitValue(match[3], "kt"))
      return fragment
    })
  })

  headings.forEach((heading) => {
    replaceTextMatches(heading, WIND_HEADER_PATTERN, () => {
      const fragment = document.createDocumentFragment()
      fragment.append(unitValue("m/s", "m/s", false))
      fragment.append(unitValue("kt", "kt", false))
      return fragment
    })
  })

  return pairs > 0
}

function preferredWindUnit() {
  try {
    return localStorage.getItem("geo2wf-wind-unit") === "kt" ? "kt" : "m/s"
  } catch (_error) {
    return "m/s"
  }
}

function setWindUnit(unit) {
  document.querySelectorAll(".datatable-wrapper[data-wind-unit]").forEach((wrapper) => {
    wrapper.dataset.windUnit = unit
  })
  document.querySelectorAll("[data-set-wind-unit]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.setWindUnit === unit))
  })
  try {
    localStorage.setItem("geo2wf-wind-unit", unit)
  } catch (_error) {
    // Unit switching still works when storage is unavailable.
  }
}

function addWindUnitControl(wrapper) {
  wrapper.dataset.windUnit = preferredWindUnit()

  const toolbar = document.createElement("div")
  toolbar.className = "table-unit-control"

  const label = document.createElement("span")
  label.className = "table-unit-control__label"
  label.textContent = "Wind units"
  toolbar.append(label)

  const buttons = document.createElement("span")
  buttons.className = "table-unit-control__buttons"
  buttons.setAttribute("role", "group")
  buttons.setAttribute("aria-label", "Wind speed units")

  ;["m/s", "kt"].forEach((unit) => {
    const button = document.createElement("button")
    button.type = "button"
    button.dataset.setWindUnit = unit
    button.textContent = unit
    button.setAttribute("aria-pressed", String(unit === preferredWindUnit()))
    button.addEventListener("click", () => setWindUnit(unit))
    buttons.append(button)
  })

  toolbar.append(buttons)
  const container = wrapper.querySelector(".datatable-container")
  wrapper.insertBefore(toolbar, container)
}

function enhanceTables() {
  if (!window.simpleDatatables?.DataTable) return

  document.querySelectorAll(".md-typeset table:not(.no-datatable)").forEach((table) => {
    if (table.dataset.enhanced === "true") return

    const rowCount = table.tBodies[0]?.rows.length ?? 0
    if (rowCount < TABLE_MIN_ROWS || !table.tHead) return

    table.dataset.enhanced = "true"
    const hasWindValues = prepareWindValues(table)
    const searchable = rowCount >= TABLE_SEARCH_MIN_ROWS
    const paging = rowCount >= TABLE_PAGE_MIN_ROWS

    const dataTable = new simpleDatatables.DataTable(table, {
      searchable,
      sortable: true,
      paging,
      perPage: TABLE_PAGE_SIZE,
      perPageSelect: [15, 25, 50, 100],
      labels: {
        placeholder: "Search table…",
        searchTitle: "Search within this table",
        perPage: "rows per page",
        noRows: "No matching rows",
        noResults: "No results match your search",
        info: "Showing {start}–{end} of {rows} rows"
      }
    })

    if (hasWindValues) addWindUnitControl(dataTable.wrapperDOM)
  })

  setWindUnit(preferredWindUnit())
}

document$.subscribe(() => {
  document.querySelectorAll("[data-current-year]").forEach((node) => {
    node.textContent = new Date().getFullYear()
  })
  enhanceTables()
})

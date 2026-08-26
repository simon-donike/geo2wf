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

function prepareSortValues(table) {
  table.querySelectorAll("tbody tr").forEach((row) => {
    Array.from(row.cells).forEach((cell) => {
      if (!cell.hasAttribute("data-order")) cell.dataset.order = cell.textContent.trim()
    })
  })
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
      const lineBreak = document.createElement("br")
      lineBreak.className = "table-heading-break"
      fragment.append(lineBreak)
      fragment.append(unitValue("m/s", "m/s", false))
      fragment.append(unitValue("kt", "kt", false))
      return fragment
    })
  })

  return pairs > 0
}

function markBestValues(table) {
  const headings = [...table.querySelectorAll("thead th")]

  headings.forEach((heading, index) => {
    const headingText = heading.textContent
    const direction = headingText.includes("↓") ? "min" : headingText.includes("↑") ? "max" : null
    if (!direction) return

    const candidates = [...table.querySelectorAll("tbody tr")]
      .map((row) => row.cells[index])
      .filter(Boolean)
      .map((cell) => {
        const match = cell.dataset.order?.match(/^[+\-\u2212]?\d+(?:\.\d+)?/u)
        return match ? { cell, value: Number(match[0].replace("\u2212", "-")) } : null
      })
      .filter((candidate) => candidate && Number.isFinite(candidate.value))

    if (!candidates.length) return
    const best = Math[direction](...candidates.map((candidate) => candidate.value))
    candidates
      .filter((candidate) => candidate.value === best)
      .forEach(({ cell }) => {
        cell.classList.add("table-best-value")
        cell.dataset.best = direction === "min" ? "lowest" : "highest"
        cell.title = `Best ${cell.dataset.best} value in this table`
      })
  })
}

function classifyTableWidth(table) {
  const columnCount = table.tHead?.rows[0]?.cells.length ?? 0
  if (columnCount >= 10) table.classList.add("datatable-table--very-wide")
  else if (columnCount >= 7) table.classList.add("datatable-table--wide")
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

  const units = ["m/s", "kt"]
  units.forEach((unit) => {
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

function enhanceTable(table) {
  if (!window.simpleDatatables?.DataTable) return

  if (table.dataset.enhanced === "true") return
  const rowCount = table.tBodies[0]?.rows.length ?? 0
  if (rowCount < TABLE_MIN_ROWS || !table.tHead) return

  table.dataset.enhanced = "true"
  prepareSortValues(table)
  const hasWindValues = prepareWindValues(table)
  markBestValues(table)
  classifyTableWidth(table)
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
}

function enhanceTables() {
  document.querySelectorAll(".md-typeset table:not(.no-datatable)").forEach(enhanceTable)

  setWindUnit(preferredWindUnit())
}

function parseCsv(text) {
  const rows = []
  let row = []
  let field = ""
  let quoted = false

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index]
    if (quoted) {
      if (character === '"' && text[index + 1] === '"') {
        field += '"'
        index += 1
      } else if (character === '"') {
        quoted = false
      } else {
        field += character
      }
    } else if (character === '"') {
      quoted = true
    } else if (character === ",") {
      row.push(field)
      field = ""
    } else if (character === "\n") {
      row.push(field.endsWith("\r") ? field.slice(0, -1) : field)
      rows.push(row)
      row = []
      field = ""
    } else {
      field += character
    }
  }

  if (field || row.length) {
    row.push(field.endsWith("\r") ? field.slice(0, -1) : field)
    rows.push(row)
  }
  return rows
}

function renderCsvTable(viewer, csvRows) {
  if (csvRows.length < 2) throw new Error("The CSV contains no data rows")

  const headers = csvRows[0]
  headers[0] = headers[0].replace(/^\uFEFF/u, "")
  const requestedColumns = viewer.dataset.csvColumns
    ? viewer.dataset.csvColumns.split(",").map((column) => column.trim())
    : headers
  const labels = viewer.dataset.csvLabels
    ? viewer.dataset.csvLabels.split("|").map((label) => label.trim())
    : requestedColumns
  if (labels.length !== requestedColumns.length) {
    throw new Error("CSV column and label counts do not match")
  }

  const indices = requestedColumns.map((column) => {
    const index = headers.indexOf(column)
    if (index === -1) throw new Error(`CSV column not found: ${column}`)
    return index
  })
  const table = document.createElement("table")
  const head = table.createTHead()
  const headingRow = head.insertRow()
  labels.forEach((label) => {
    const cell = document.createElement("th")
    cell.scope = "col"
    cell.textContent = label
    headingRow.append(cell)
  })

  const body = table.createTBody()
  csvRows.slice(1).forEach((csvRow) => {
    if (csvRow.length === 1 && !csvRow[0]) return
    const tableRow = body.insertRow()
    indices.forEach((index) => {
      const cell = tableRow.insertCell()
      cell.textContent = csvRow[index] || viewer.dataset.csvEmpty || ""
    })
  })

  viewer.replaceChildren(table)
  enhanceTable(table)
  setWindUnit(preferredWindUnit())
}

function loadCsvTables() {
  document.querySelectorAll("[data-csv-source]").forEach(async (viewer) => {
    if (viewer.dataset.csvLoaded === "true") return
    viewer.dataset.csvLoaded = "true"
    try {
      const response = await fetch(new URL(viewer.dataset.csvSource, document.baseURI))
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      renderCsvTable(viewer, parseCsv(await response.text()))
    } catch (error) {
      viewer.dataset.csvLoaded = "false"
      const status = viewer.querySelector(".csv-table-viewer__status")
      if (status) status.textContent = "The observation table could not be loaded. Use the CSV download above."
      console.error("Could not load CSV table", error)
    }
  })
}

document$.subscribe(() => {
  document.querySelectorAll("[data-current-year]").forEach((node) => {
    node.textContent = new Date().getFullYear()
  })
  loadCsvTables()
  enhanceTables()
})

window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
}

document$.subscribe(() => {
  if (typeof MathJax.typesetPromise === "function") {
    MathJax.typesetPromise()
  }
})

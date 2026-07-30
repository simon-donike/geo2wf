---
hide:
  - toc
---

# Interactive storm explorer

The storm explorer presents the `AL082025` and `EP112025` inference sequences as complete, interactive storm tracks. Follow each GOES-conditioned prediction through time and compare five wind-field diagnostics with the available matched SAR observations.

[Open the storm explorer :material-open-in-new:](explorer/dashboard.html){ .md-button .md-button--primary }

!!! info "How to read the charts"
    Solid red lines show metrics derived from geo2wf predictions. Gray dots are matched SAR observations; dashed gray segments interpolate visually between those sparse measurements.

The explorer is generated as plain HTML, CSS, JavaScript, and compact JSON, so it deploys with this documentation and needs no server or API key.

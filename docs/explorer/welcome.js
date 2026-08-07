const welcomeBanner = document.querySelector("#welcomeBanner");
const welcomeDismissButtons = document.querySelectorAll("#dismissWelcome, #closeWelcome");
const startGuidedTour = document.querySelector("#startGuidedTour");

if (welcomeBanner && !welcomeBanner.open) {
  if (typeof welcomeBanner.showModal === "function") {
    welcomeBanner.showModal();
  } else {
    welcomeBanner.setAttribute("open", "");
  }
}

welcomeDismissButtons.forEach((button) => button.addEventListener("click", () => {
  if (typeof welcomeBanner.close === "function") {
    welcomeBanner.close();
  } else {
    welcomeBanner.removeAttribute("open");
  }
}));

const tourLayer = document.querySelector("#guidedTour");
const tourCard = document.querySelector("#tourCard");
const tourSpotlight = document.querySelector("#tourSpotlight");
const tourTitle = document.querySelector("#tourTitle");
const tourDescription = document.querySelector("#tourDescription");
const tourKicker = document.querySelector("#tourKicker");
const tourProgress = document.querySelector("#tourProgress");
const tourDots = document.querySelector("#tourDots");
const tourBack = document.querySelector("#tourBack");
const tourNext = document.querySelector("#tourNext");
const closeTour = document.querySelector("#closeTour");
let tourIndex = 0;
let activeTourTarget = null;

const wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
const visible = (selector) => [...document.querySelectorAll(selector)].find((element) => !element.hidden && element.getClientRects().length);
const setTourMode = async (mode) => {
  const button = document.querySelector(mode === "forecast" ? "#forecastMode" : "#nowcastMode");
  if (button && !button.disabled && button.getAttribute("aria-selected") !== "true") button.click();
  await wait(mode === "forecast" ? 850 : 80);
};
const focusRapidIntensification = async () => {
  window.StormSenseTour?.focusRapidIntensification?.();
  await wait(80);
};

const tourSteps = [
  {
    kicker: "Our central challenge",
    title: "Rapid intensification, made visible",
    description: "Rapid intensification—an IBTrACS wind increase of at least 30 knots in 24 hours—is the event we care about most. We are especially proud that our nowcasting and forecasting results retain this fast-changing signal. StormSense places hatched RI periods behind every wind chart so you can inspect that performance directly.",
    target: () => visible("#charts .chart-card:first-child") || document.querySelector(".analytics"),
    enter: async () => { await setTourMode("nowcast"); await focusRapidIntensification(); }
  },
  {
    kicker: "Choose the question",
    title: "Storm and analysis mode",
    description: "Mode switches between Nowcast—an estimate at the observation time—and Forecast—a prediction verified 12 hours later. Storm selects the tropical-cyclone track you want to investigate.",
    target: () => document.querySelector(".map-controls")
  },
  {
    kicker: "Explore the track",
    title: "Map navigation",
    description: "Drag to pan, scroll or pinch to zoom, and use the +/− buttons for precise zooming. The highlighted marker follows the selected timeline observation along the complete storm track.",
    target: () => document.querySelector("#map")
  },
  {
    kicker: "Build the visual stack",
    title: "Imagery visibility and drawing order",
    description: "The checkboxes show or hide geostationary, PMW, and SAR imagery. Drag the dotted handles to change their drawing order—the top item is drawn above the others. Arrow keys also move a focused handle.",
    target: () => document.querySelector(".leaflet-control-layers")
  },
  {
    kicker: "Move through the event",
    title: "Timeline and animation",
    description: "Play starts or pauses the sequence. Drag the time slider to inspect an exact observation. Animation limits imagery to the current frame, while Speed controls playback from 0.5× to 4×. The date and observation counter update together.",
    target: () => document.querySelector(".timeline")
  },
  {
    kicker: "Read the current state",
    title: "Observation summary",
    description: "These cards identify the selected storm, count the available geostationary frames, SAR matches, and PMW swaths, and report the IBTrACS intensity class at the current time.",
    target: () => document.querySelector(".summary-strip")
  },
  {
    kicker: "Nowcasting",
    title: "Compare inference models",
    description: "The inference selector changes the highlighted nowcast curve. UNet+MLP supplies its corrected maximum-wind estimate while its upstream UNet supplies spatial diagnostics; the other available models expose their own wind-field diagnostics.",
    target: () => visible(".model-toolbar"),
    enter: async () => { await setTourMode("nowcast"); await focusRapidIntensification(); }
  },
  {
    kicker: "Nowcasting rapid intensification",
    title: "Following the sharp rise now",
    description: "The colored prediction curve can be compared directly with SAR-derived wind and the cream IBTrACS reference. We are especially proud of how closely the nowcast follows the sharp upward structure inside the hatched RI interval instead of smoothing away the event. Hover anywhere on a chart for exact values and classification.",
    target: () => visible("#charts .chart-card:first-child"),
    enter: focusRapidIntensification
  },
  {
    kicker: "Add context",
    title: "NWP and post-processing controls",
    description: "NWP forecasts adds available full-track weather-model runs as muted dashed curves. Post-processing applies a six-hour centered median to the selected nowcast, letting you compare a steadier signal with the raw model response.",
    target: () => visible(".graph-toolbar"),
    enter: async () => { await setTourMode("nowcast"); }
  },
  {
    kicker: "Forecasting",
    title: "Look 12 hours ahead",
    description: "Forecast changes the charts from current-time inference to retrospective +12-hour prediction. Map time becomes issue time; the highlighted chart target is the valid time 12 hours later.",
    target: () => document.querySelector(".mode-switch"),
    enter: async () => { await setTourMode("forecast"); await focusRapidIntensification(); }
  },
  {
    kicker: "Forecast configuration",
    title: "Model and lead time",
    description: "Choose among the available forecast models here. The lead badge states how far ahead each prediction is evaluated, so model output and the later IBTrACS observation are compared at the same valid time. If forecast data cannot load, the notice provides a Retry control.",
    target: () => visible("#forecastToolbar") || document.querySelector(".mode-switch"),
    enter: async () => { await setTourMode("forecast"); }
  },
  {
    kicker: "Forecasting rapid intensification",
    title: "Anticipating the rise",
    description: "The forecast curve is plotted against its later IBTrACS verification inside the same hatched RI window. We are especially proud that the model captures the rapid rise 12 hours ahead. The issue line, valid-time line, and connecting lead band make that result directly inspectable.",
    target: () => visible("#charts .chart-card:first-child") || document.querySelector(".analytics"),
    enter: async () => { await setTourMode("forecast"); await focusRapidIntensification(); }
  },
  {
    kicker: "Decode the evidence",
    title: "Legends and optional comparisons",
    description: "The active legend identifies model prediction, IBTrACS verification, RI hatching, valid-time behavior, and optional NWP runs. Toggle NWP forecasts below the charts whenever you want that broader baseline comparison.",
    target: () => visible("#forecastLegend") || visible("#nowcastLegend")
  },
  {
    kicker: "Keep exploring",
    title: "Help, status, and documentation",
    description: "Inference data confirms the dashboard is using model output. The ? button explains methods and visual encodings, while Documentation opens the deeper scientific and technical reference. You can restart this tour from the welcome screen after reloading the page.",
    target: () => document.querySelector(".top-actions")
  }
];

function positionTour() {
  if (tourLayer.hidden || !activeTourTarget) return;
  const targetRect = activeTourTarget.getBoundingClientRect();
  const padding = 7;
  tourSpotlight.style.left = `${Math.max(4, targetRect.left - padding)}px`;
  tourSpotlight.style.top = `${Math.max(4, targetRect.top - padding)}px`;
  tourSpotlight.style.width = `${Math.min(innerWidth - 8, targetRect.width + padding * 2)}px`;
  tourSpotlight.style.height = `${Math.min(innerHeight - 8, targetRect.height + padding * 2)}px`;

  tourCard.style.left = "12px";
  tourCard.style.top = "12px";
  tourCard.style.bottom = "auto";
  const cardRect = tourCard.getBoundingClientRect();
  const gap = 18;
  if (innerWidth <= 700) {
    tourCard.style.left = `${Math.max(10, (innerWidth - cardRect.width) / 2)}px`;
    tourCard.style.top = "auto";
    tourCard.style.bottom = "10px";
    return;
  }
  let left;
  let top;
  if (innerWidth - targetRect.right >= cardRect.width + gap) {
    left = targetRect.right + gap;
    top = targetRect.top;
  } else if (targetRect.left >= cardRect.width + gap) {
    left = targetRect.left - cardRect.width - gap;
    top = targetRect.top;
  } else if (innerHeight - targetRect.bottom >= cardRect.height + gap) {
    left = targetRect.left + targetRect.width / 2 - cardRect.width / 2;
    top = targetRect.bottom + gap;
  } else {
    left = targetRect.left + targetRect.width / 2 - cardRect.width / 2;
    top = targetRect.top - cardRect.height - gap;
  }
  tourCard.style.left = `${Math.max(10, Math.min(innerWidth - cardRect.width - 10, left))}px`;
  tourCard.style.top = `${Math.max(10, Math.min(innerHeight - cardRect.height - 10, top))}px`;
}

async function showTourStep(index, focusCard = true) {
  tourIndex = Math.max(0, Math.min(tourSteps.length - 1, index));
  const step = tourSteps[tourIndex];
  tourBack.disabled = true;
  tourNext.disabled = true;
  activeTourTarget?.classList.remove("tour-active-target");
  await step.enter?.();
  activeTourTarget = step.target?.() || document.querySelector(".workspace");
  activeTourTarget.classList.add("tour-active-target");
  tourKicker.textContent = step.kicker;
  tourTitle.textContent = step.title;
  tourDescription.textContent = step.description;
  tourProgress.textContent = `StormSense tour · ${tourIndex + 1} of ${tourSteps.length}`;
  tourDots.innerHTML = tourSteps.map((_, i) => `<i class="${i === tourIndex ? "active" : ""}"></i>`).join("");
  tourBack.disabled = tourIndex === 0;
  tourNext.disabled = false;
  tourNext.textContent = tourIndex === tourSteps.length - 1 ? "Explore on your own" : "Next";
  await new Promise(requestAnimationFrame);
  positionTour();
  if (focusCard) tourCard.focus({ preventScroll: true });
}

function endTour() {
  activeTourTarget?.classList.remove("tour-active-target");
  activeTourTarget = null;
  tourLayer.hidden = true;
  document.body.classList.remove("tour-running");
  document.querySelector("#helpButton")?.focus({ preventScroll: true });
}

async function beginTour() {
  if (typeof welcomeBanner?.close === "function") welcomeBanner.close();
  else welcomeBanner?.removeAttribute("open");
  while (!window.StormSenseTour && !document.querySelector("#loading.hidden")) await wait(80);
  tourLayer.hidden = false;
  document.body.classList.add("tour-running");
  await showTourStep(0);
}

startGuidedTour?.addEventListener("click", beginTour);
tourBack?.addEventListener("click", () => showTourStep(tourIndex - 1));
tourNext?.addEventListener("click", () => tourIndex === tourSteps.length - 1 ? endTour() : showTourStep(tourIndex + 1));
closeTour?.addEventListener("click", endTour);
window.addEventListener("resize", positionTour);
window.addEventListener("keydown", (event) => {
  if (tourLayer?.hidden) return;
  if (event.key === "Escape") endTour();
  if (event.key === "Tab") {
    const controls = [...tourCard.querySelectorAll("button:not(:disabled)")];
    const first = controls[0];
    const last = controls[controls.length - 1];
    if (event.shiftKey && (document.activeElement === first || document.activeElement === tourCard)) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  }
  if (event.key === "ArrowRight" && !event.target.closest?.("select, input")) showTourStep(tourIndex + 1);
  if (event.key === "ArrowLeft" && !event.target.closest?.("select, input")) showTourStep(tourIndex - 1);
});

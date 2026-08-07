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
    kicker: "Why StormSense",
    title: "Rapid intensification, captured",
    description: "Hatching marks an IBTrACS rise of at least 30 knots in 24 hours. We are proud that both our nowcasts and forecasts retain this sharp signal.",
    target: () => visible("#charts .chart-card:first-child") || document.querySelector(".analytics"),
    enter: async () => { await setTourMode("nowcast"); await focusRapidIntensification(); }
  },
  {
    kicker: "Choose a view",
    title: "Storm and mode",
    description: "Storm chooses the track. Mode switches between inference at observation time and a prediction verified 12 hours later.",
    target: () => document.querySelector(".map-controls")
  },
  {
    kicker: "Explore the evidence",
    title: "Map and imagery stack",
    description: "Pan or zoom the map. Use the imagery checkboxes for visibility, then drag the dotted grips—the top layer draws above the others.",
    target: () => document.querySelector(".leaflet-control-layers")
  },
  {
    kicker: "Move through time",
    title: "Timeline and storm summary",
    description: "Play, scrub, animate, and set playback speed here. The summary cards report available imagery and the current IBTrACS class.",
    target: () => document.querySelector(".timeline")
  },
  {
    kicker: "Nowcasting",
    title: "Inspect the rise as it happens",
    description: "Choose an inference model, then compare its curve with SAR and IBTrACS. Hover the chart for exact values inside the RI interval.",
    target: () => document.querySelector(".analytics"),
    enter: async () => { await setTourMode("nowcast"); await focusRapidIntensification(); }
  },
  {
    kicker: "Add context",
    title: "NWP and post-processing",
    description: "NWP adds full-track forecast baselines. Post-processing applies a six-hour median to the selected nowcast.",
    target: () => visible(".graph-toolbar"),
    enter: async () => { await setTourMode("nowcast"); }
  },
  {
    kicker: "Forecasting",
    title: "Anticipate rapid intensification",
    description: "Select a forecast model and compare its +12-hour curve with later IBTrACS verification. Issue and valid-time markers expose the lead directly.",
    target: () => document.querySelector(".analytics"),
    enter: async () => { await setTourMode("forecast"); await focusRapidIntensification(); }
  },
  {
    kicker: "Keep exploring",
    title: "Legends, help, and documentation",
    description: "Legends decode every line and RI hatch. Use ? for methodology, Retry if a forecast fails, and Documentation for the full reference.",
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

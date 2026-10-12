export type MethodView = "inference" | "training" | "pipeline";
export interface ShowcaseStep {
  id: string;
  title: string;
  caption: string;
  anchor: string;
  page: "overview" | "storm" | "method";
  time?: string;
  view?: MethodView;
  mode?: "joint" | "encoder";
}
export const SHOWCASE_STEPS: ShowcaseStep[] = [
  {
    id: "live",
    title: "Live inference",
    page: "overview",
    anchor: "live-overview",
    caption: "Satellite scans → server-side inference → published winds. Hourly updates while the runner is online.",
  },
  {
    id: "storm",
    title: "Explore a storm",
    page: "storm",
    anchor: "storm-map",
    time: "2026-09-27T06:00:00Z",
    caption:
      "Now a saved example: Nolo’s full history. Drag or play the timeline.",
  },
  {
    id: "reference",
    title: "Two routes to wind estimates",
    page: "storm",
    anchor: "reference-method",
    time: "2026-09-27T06:00:00Z",
    caption:
      "Reference: multiple observing systems. StormSense: infrared imagery + context, without reconnaissance inputs at inference.",
  },
  {
    id: "inputs",
    title: "Model inputs",
    page: "method",
    anchor: "inputs",
    view: "inference",
    mode: "joint",
    caption:
      "Ian’s saved example: infrared bands and storm context. GeoColor is display-only.",
  },
  {
    id: "network",
    title: "Inside the network",
    page: "method",
    anchor: "network",
    view: "inference",
    mode: "joint",
    caption:
      "Shared features feed wind-field and scalar prediction branches.",
  },
  {
    id: "outputs",
    title: "Read the outputs",
    page: "method",
    anchor: "outputs",
    view: "inference",
    mode: "joint",
    caption:
      "Maximum wind estimates intensity; wind fields and radii describe structure.",
  },
  {
    id: "supervision",
    title: "Training targets",
    page: "method",
    anchor: "supervision",
    view: "training",
    mode: "joint",
    caption:
      "SAR winds and observed storm properties supervise training—not inference inputs.",
  },
  {
    id: "comparison",
    title: "Encoder-only model",
    page: "method",
    anchor: "network",
    view: "inference",
    mode: "encoder",
    caption:
      "No field decoder. Direct wind and radius estimates from a separate checkpoint.",
  },
  {
    id: "pipeline",
    title: "Publication pipeline",
    page: "method",
    anchor: "pipeline",
    view: "pipeline",
    mode: "joint",
    caption:
      "Retrieve → preprocess → infer → publish. Display imagery follows a separate path.",
  },
  {
    id: "interpretation",
    title: "Compare with observations",
    page: "storm",
    anchor: "storm-results",
    time: "2026-09-28T06:00:00Z",
    caption:
      "Compare Nolo’s hindcasts with reference winds. Inspect timestamps and provenance.",
  },
];

export function showcaseUrl(step: ShowcaseStep) {
  const params = new URLSearchParams({ showcase: "v1", tour: step.id });
  if (step.time) params.set("time", step.time);
  if (step.view) params.set("view", step.view);
  if (step.mode) params.set("model", step.mode);
  return `${step.page === "overview" ? "/" : step.page === "storm" ? "/storms/EP152026" : "/method"}?${params}`;
}

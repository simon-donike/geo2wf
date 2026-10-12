import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  Link,
  NavLink,
  useLocation,
  useNavigate,
  type LinkProps,
  type NavLinkProps,
} from "react-router-dom";
import { DataSourceContext } from "./dataContext";
import { LIVE_DATA } from "./data";
import { preloadShowcase, SHOWCASE_DATA } from "./showcaseData";
import {
  SHOWCASE_STEPS,
  showcaseUrl,
  type ShowcaseStep,
} from "./showcaseSteps";
import ShowcaseHighlight from "./ShowcaseHighlight";
import "./showcase.css";

const preferenceKeys = [
  "stormsense-unit",
  "stormsense-smoothing",
  "stormsense-satellite",
];
const returnKey = "stormsense-showcase-return";
interface ReturnState {
  url: string;
  preferences: Record<string, string | null>;
}
function capture(url: string) {
  sessionStorage.setItem(
    returnKey,
    JSON.stringify({
      url,
      preferences: Object.fromEntries(
        preferenceKeys.map((key) => [key, localStorage.getItem(key)]),
      ),
    }),
  );
  localStorage.setItem("stormsense-unit", "kt");
  localStorage.setItem("stormsense-smoothing", "off");
  localStorage.setItem("stormsense-satellite", "on");
}
function restore() {
  let saved: ReturnState | null = null;
  try {
    saved = JSON.parse(sessionStorage.getItem(returnKey) || "null");
  } catch {
    /* Invalid old session. */
  }
  if (saved?.preferences)
    for (const key of preferenceKeys) {
      const value = saved.preferences[key];
      if (typeof value === "string") localStorage.setItem(key, value);
      else localStorage.removeItem(key);
    }
  sessionStorage.removeItem(returnKey);
  return saved?.url?.startsWith("/") && !saved.url.startsWith("//")
    ? saved.url
    : "/method";
}

interface TourContext {
  active: boolean;
  step: ShowcaseStep;
  entry: string;
  start: () => void;
  link: (to: string) => string;
}
const ShowcaseContext = createContext<TourContext>({
  active: false,
  step: SHOWCASE_STEPS[0],
  entry: "",
  start: () => {},
  link: (to) => to,
});
export const useShowcase = () => useContext(ShowcaseContext);

/** Ordinary in-app navigation keeps the fixed data while visitors explore. */
export function ShowcaseLink({ to, ...props }: LinkProps) {
  const tour = useShowcase();
  return <Link {...props} to={typeof to === "string" ? tour.link(to) : to} />;
}
export function ShowcaseNavLink({ to, ...props }: NavLinkProps) {
  const tour = useShowcase();
  return (
    <NavLink {...props} to={typeof to === "string" ? tour.link(to) : to} />
  );
}

export default function ShowcaseProvider({
  children,
}: {
  children: ReactNode;
}) {
  const location = useLocation();
  const navigate = useNavigate();
  const params = new URLSearchParams(location.search);
  const active = params.get("showcase") === "v1";
  const index = Math.max(
    0,
    SHOWCASE_STEPS.findIndex((s) => s.id === (params.get("tour") === "learning" ? "supervision" : params.get("tour"))),
  );
  const step = SHOWCASE_STEPS[index];
  const bundled = active && step.page !== "overview";
  const [revision, setRevision] = useState(0);
  const [entry, setEntry] = useState(0);
  const [status, setStatus] = useState<"loading" | "ready" | "error">(
    "loading",
  );
  const [error, setError] = useState("");
  const [progress, setProgress] = useState([0, 0]);
  const [preferenceEpoch, setPreferenceEpoch] = useState(0);
  const wasActive = useRef(false);
  const heading = useRef<HTMLHeadingElement>(null);
  const entryKey = `${step.id}:${entry}`;

  useEffect(() => {
    if (!active) {
      if (wasActive.current && sessionStorage.getItem(returnKey)) {
        restore();
        setPreferenceEpoch((n) => n + 1);
      }
      wasActive.current = false;
      return;
    }
    wasActive.current = true;
    if (!sessionStorage.getItem(returnKey)) capture("/method");
    const controller = new AbortController();
    setStatus("loading");
    setError("");
    void Promise.all([
      preloadShowcase(controller.signal, (done, total) => {
        if (!controller.signal.aborted) setProgress([done, total]);
      }),
      import("./Method"),
    ])
      .then(() => {
        if (!controller.signal.aborted) setStatus("ready");
      })
      .catch((cause) => {
        if (controller.signal.aborted) return;
        setError(String(cause.message || cause));
        setStatus("error");
        controller.abort();
      });
    return () => controller.abort();
  }, [active, revision]);

  // A direct stop link is canonicalized once; subsequent scrubbing remains free.
  useEffect(() => {
    if (!active) return;
    if (location.pathname + location.search !== showcaseUrl(step))
      navigate(showcaseUrl(step), { replace: true });
  }, [active, step.id]);

  const go = (next: number) => {
    setEntry((n) => n + 1);
    navigate(showcaseUrl(SHOWCASE_STEPS[next]));
  };
  const start = () => {
    capture(location.pathname + location.search + location.hash);
    go(0);
  };
  const exit = () => {
    const url = restore();
    setPreferenceEpoch((n) => n + 1);
    navigate(url, { replace: true });
  };
  const link = (to: string) => {
    if (!active || !to.startsWith("/") || to.startsWith("//")) return to;
    const url = new URL(to, window.location.origin);
    url.searchParams.set("showcase", "v1");
    url.searchParams.set("tour", step.id);
    url.searchParams.delete("tour_paused");
    return url.pathname + url.search + url.hash;
  };
  return (
    <ShowcaseContext.Provider
      value={{ active, step, entry: entryKey, start, link }}
    >
      <DataSourceContext.Provider value={bundled ? SHOWCASE_DATA : LIVE_DATA}>
        <div
          className={active ? "showcase-shell" : undefined}
          data-showcase-step={active ? step.id : undefined}
        >
          <div
            id="showcase-content"
            className={active ? "showcase-content" : undefined}
            key={`${active}:${bundled}:${preferenceEpoch}`}
          >
            {!bundled || status === "ready" ? (
              children
            ) : (
              <main
                className="showcase-preparation"
                aria-label="Showcase preparation"
              >
                <span className="eyebrow">
                  StormSense · guided tour
                </span>
                <h1>
                  {status === "error"
                    ? "The showcase needs an asset."
                    : "Preparing your showcase."}
                </h1>
                <p role={status === "error" ? "alert" : "status"}>
                  {status === "error"
                    ? error
                    : `Checking the fixed storm record and images · ${progress[0]} / ${progress[1]}`}
                </p>
                {status === "error" && (
                  <button onClick={() => setRevision((n) => n + 1)}>
                    Retry preparation
                  </button>
                )}
              </main>
            )}
          </div>
          {active && (!bundled || status === "ready") && (
            <ShowcaseHighlight anchor={step.anchor} entry={entryKey} page={step.page} heading={heading} />
          )}
          {active && (
            <aside className="showcase-controls" aria-label="Showcase controls" data-ready={!bundled || status === "ready"} data-step={step.id} data-assets-ready={status === "ready"}>
              <div className="showcase-caption">
                <span className="eyebrow">
                  Guided tour · 
                  {index + 1} / {SHOWCASE_STEPS.length}
                </span>
                <h2 ref={heading} tabIndex={-1}>
                  {step.title}
                </h2>
                <p aria-live="polite">
                  {step.caption}
                </p>
              </div>
              {!bundled && status === "error" && (
                <p role="alert" className="showcase-asset-error">
                  Saved example unavailable. <button onClick={() => setRevision(n => n + 1)}>Retry preparation</button>
                </p>
              )}
              <div className="showcase-navigation">
                <div className="showcase-buttons">
                  <button
                    disabled={!index || (bundled && status !== "ready")}
                    onClick={() => go(index - 1)}
                  >
                    Back
                  </button>
                  <button
                    className="showcase-primary"
                    disabled={index === SHOWCASE_STEPS.length - 1 || status !== "ready"}
                    onClick={() => go(index + 1)}
                  >
                    Next
                  </button>
                  <button onClick={exit}>Exit</button>
                </div>

              </div>
            </aside>
          )}
        </div>
      </DataSourceContext.Provider>
    </ShowcaseContext.Provider>
  );
}

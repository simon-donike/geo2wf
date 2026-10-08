import { useEffect, useRef, useState, type CSSProperties } from "react";
import exampleData from "../public/method/example.json";
import type { MethodExample, MethodMode } from "./methodExample";
import { stamp, wind } from "./data";
import "./method.css";
import MethodTraining from "./MethodTraining";

const example: MethodExample = exampleData;
const vars = (values: Record<string, string | number>) =>
  values as CSSProperties;

function useFlowMotion() {
  const container = useRef<HTMLDivElement>(null);
  const [reduced, setReduced] = useState(
    () => window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  const [paused, setPaused] = useState(false);
  const [visible, setVisible] = useState(false);
  const [foreground, setForeground] = useState(!document.hidden);
  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const change = () => setReduced(media.matches);
    const visibility = () => setForeground(!document.hidden);
    media.addEventListener("change", change);
    document.addEventListener("visibilitychange", visibility);
    const observer = new IntersectionObserver(
      ([entry]) => setVisible(entry.isIntersecting),
      { threshold: 0.05 },
    );
    if (container.current) observer.observe(container.current);
    return () => {
      observer.disconnect();
      media.removeEventListener("change", change);
      document.removeEventListener("visibilitychange", visibility);
    };
  }, []);
  return {
    container,
    reduced,
    paused,
    setPaused,
    motion: reduced
      ? "reduced"
      : paused || !visible || !foreground
        ? "paused"
        : "running",
  };
}

function Flow({
  d,
  delay = 0,
  tone = "teal",
}: {
  d: string;
  delay?: number;
  tone?: string;
}) {
  return (
    <g className={`method-flow ${tone}`}>
      <path d={d} className="method-flow-base" />
      <path
        d={d}
        pathLength={1}
        strokeDasharray=".16 .84"
        className="method-flow-lit"
        style={vars({ "--flow-delay": `${delay}s` })}
      />
    </g>
  );
}

interface StackLayer {
  id: string;
  image: string;
  label: string;
  short: string;
}
function ImageStack({
  label,
  layers,
  initial = 0,
  context = false,
}: {
  label: string;
  layers: StackLayer[];
  initial?: number;
  context?: boolean;
}) {
  const [selected, setSelected] = useState(initial);
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const [pinned, setPinned] = useState(false);
  const expanded = hovered || focused || pinned;
  return (
    <div
      className={`method-stack-card ${context ? "context-stack" : "satellite-stack"}`}
    >
      <div className="method-stack-label">
        <span>{label}</span>
        <small>{layers.length} channels</small>
      </div>
      <button
        type="button"
        className="method-stack"
        aria-label={`${label} image stack`}
        aria-expanded={expanded}
        data-expanded={expanded}
        onPointerEnter={(event) => {
          if (event.pointerType === "mouse") setHovered(true);
        }}
        onPointerLeave={() => setHovered(false)}
        onFocus={(event) =>
          setFocused(event.currentTarget.matches(":focus-visible"))
        }
        onBlur={() => setFocused(false)}
        onClick={() => setPinned((value) => !value)}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            setPinned(false);
            setFocused(false);
            setHovered(false);
          }
        }}
        title="Hover, focus, or tap to expand the tensor stack"
        style={vars({ "--fan": expanded ? 1 : 0 })}
      >
        {layers.map((layer, i) => (
          <img
            key={layer.id}
            src={layer.image}
            alt={
              i === selected
                ? `Actual ${layer.label} ${context ? "input tensor" : "infrared crop"} for Hurricane Ian`
                : ""
            }
            aria-hidden={i !== selected}
            className={i === selected ? "is-front" : ""}
            style={vars({
              "--layer": i === selected ? layers.length : i,
              "--depth": i === selected ? 0 : layers.length - i,
              "--offset":
                (i === selected
                  ? 0
                  : layers.length - i - (i < selected ? 1 : 0)) /
                (layers.length - 1),
            })}
          />
        ))}
        {!context && (
          <div className="method-crop-guides" aria-hidden="true">
            <i />
            <i />
            <i />
            <i />
            <span>+</span>
          </div>
        )}
        <span className="method-image-tag">{layers[selected].label}</span>
      </button>
      <div
        className="method-band-picker"
        role="group"
        aria-label={`${label} channel`}
      >
        {layers.map((layer, i) => (
          <button
            key={layer.id}
            type="button"
            aria-label={`View ${layer.label}`}
            title={layer.label}
            aria-pressed={selected === i}
            onClick={() => setSelected(i)}
          >
            {layer.short}
          </button>
        ))}
      </div>
    </div>
  );
}

function FeatureBlock({
  x,
  y,
  size,
  lit,
  decoder = false,
}: {
  x: number;
  y: number;
  size: number;
  lit: number;
  decoder?: boolean;
}) {
  return (
    <g
      className={`method-feature ${decoder ? "decoder" : ""}`}
      style={vars({ "--lit": lit })}
    >
      <path
        d={`M${x},${y} l16,-10 h25 v${size} l-16,10 Z`}
        className="method-feature-side"
      />
      <rect
        x={x}
        y={y}
        width={25}
        height={size}
        rx={2}
        className="method-feature-front"
      />
      {[0.2, 0.4, 0.6, 0.8].map((f, i) => (
        <path
          key={i}
          d={`M${x + 4},${y + size * f} h17`}
          opacity={0.15 + lit * (0.3 + i * 0.08)}
        />
      ))}
    </g>
  );
}

function GradientFlow({ d, skip = false }: { d: string; skip?: boolean }) {
  return (
    <g
      className={`method-backward-flow ${skip ? "is-skip" : ""}`}
      aria-hidden="true"
    >
      <path
        d={d}
        className="method-backward-base"
        markerEnd="url(#method-gradient-arrow)"
      />
      <path
        d={d}
        className="method-backward-pulse"
        pathLength={1}
        strokeDasharray=".06 .19"
      />
    </g>
  );
}

function Network({
  mode,
  training = false,
}: {
  mode: MethodMode;
  training?: boolean;
}) {
  return (
    <svg
      className={`method-network ${training ? "is-training" : ""}`}
      viewBox="0 0 510 420"
      role="img"
      aria-label={
        mode === "joint"
          ? "Shared encoder branches into a field decoder with skip connections and a pooled scalar MLP"
          : "Encoder-only architecture sends pooled features directly to the scalar MLP"
      }
    >
      {training && (
        <defs>
          <marker
            id="method-gradient-arrow"
            viewBox="0 0 10 10"
            refX="8"
            refY="5"
            markerWidth="5"
            markerHeight="5"
            orient="auto-start-reverse"
          >
            <path d="M0 0 L10 5 L0 10Z" fill="#f1ad80" />
          </marker>
        </defs>
      )}
      <text x="25" y="36" className="method-svg-heading">
        SHARED ENCODER
      </text>
      <text x="25" y="54" className="method-svg-note">
        Cloud patterns → spatial features
      </text>
      <Flow d="M0 116 H30" delay={0} />
      <Flow d="M56 126 C83 126 80 177 101 177" delay={0.3} />
      <Flow d="M126 182 C155 182 151 216 178 216" delay={0.6} />
      <Flow d="M203 222 C225 222 225 247 243 247" delay={0.9} />
      <FeatureBlock x={30} y={75} size={100} lit={1} />
      <FeatureBlock x={102} y={142} size={78} lit={1} />
      <FeatureBlock x={178} y={190} size={60} lit={1} />
      <FeatureBlock x={243} y={226} size={43} lit={1} />
      <text x="257" y="294" textAnchor="middle" className="method-svg-note">
        Shared features
      </text>
      <g
        className={`method-decoder ${mode === "encoder" ? "is-retracted" : ""}`}
        aria-hidden={mode === "encoder"}
      >
        <text x="327" y="36" className="method-svg-heading method-amber">
          FIELD DECODER
        </text>
        <text x="327" y="54" className="method-svg-note">
          Features → surface wind
        </text>
        {[
          ["M58 89 C170 -5 364 -5 454 85", 0.13],
          ["M130 153 C208 85 324 85 386 149", 0.08],
          ["M206 200 C248 164 306 164 322 198", 0.03],
        ].map(([d]) => (
          <g key={d}>
            <path d={String(d)} className="method-skip-base" />
            <path
              d={String(d)}
              className="method-skip"
              style={{
                opacity: 0.65,
              }}
            />
          </g>
        ))}
        <text x="260" y="96" textAnchor="middle" className="method-svg-note">
          skip connections
        </text>
        <Flow d="M269 247 C295 247 295 217 319 217" delay={1.2} tone="amber" />
        <Flow d="M347 216 C368 216 358 180 386 180" delay={1.5} tone="amber" />
        <Flow d="M412 174 C439 174 427 123 454 123" delay={1.8} tone="amber" />
        <Flow d="M479 120 H510" delay={2.1} tone="amber" />
        <FeatureBlock x={320} y={189} size={60} lit={1} decoder />
        <FeatureBlock x={386} y={138} size={78} lit={1} decoder />
        <FeatureBlock x={454} y={72} size={100} lit={1} decoder />
      </g>
      {mode === "encoder" && (
        <g className="method-encoder-note">
          <text x="376" y="163" textAnchor="middle" className="method-svg-note">
            The scalar variant omits
          </text>
          <text x="376" y="180" textAnchor="middle" className="method-svg-note">
            the field decoder.
          </text>
        </g>
      )}
      <Flow d="M257 270 V321 Q257 339 279 339 H307" delay={1.2} />
      <rect
        x="299"
        y="317"
        width="60"
        height="45"
        rx="5"
        className="method-pooling"
      />
      <text x="329" y="337" textAnchor="middle" className="method-svg-small">
        Latent
      </text>
      <text x="329" y="352" textAnchor="middle" className="method-svg-small">
        pooling
      </text>
      <Flow d="M359 339 H378" delay={1.5} />
      <g className="method-neurons">
        {[316, 332, 348, 364].flatMap((y) =>
          [323, 340, 357].map((y2) => (
            <line key={`${y}-${y2}`} x1="382" y1={y} x2="418" y2={y2} />
          )),
        )}
        {[323, 340, 357].flatMap((y) =>
          [330, 351].map((y2) => (
            <line key={`${y}-${y2}`} x1="418" y1={y} x2="454" y2={y2} />
          )),
        )}
        {[316, 332, 348, 364].map((y) => (
          <circle key={y} cx="382" cy={y} r="4" />
        ))}
        {[323, 340, 357].map((y) => (
          <circle key={y} cx="418" cy={y} r="4" />
        ))}
        {[330, 351].map((y) => (
          <circle key={y} cx="454" cy={y} r="4" />
        ))}
      </g>
      <Flow d="M460 340 H510" delay={1.8} />
      {training && (
        <g
          className="method-backpropagation"
          aria-label="Gradients flow from losses through the output branches back into the shared encoder"
        >
          {mode === "joint" && (
            <>
              <GradientFlow d="M508 120 H467 V123 C440 123 440 177 399 177 C366 177 366 217 332 217 C295 217 295 247 257 247" />
              <GradientFlow d="M454 85 C364 -5 170 -5 58 89" skip />
              <GradientFlow d="M386 149 C324 85 208 85 130 153" skip />
              <GradientFlow d="M322 198 C306 164 248 164 206 200" skip />
            </>
          )}
          <GradientFlow d="M508 340 H382 L359 339 H279 Q257 339 257 321 V263" />
          <GradientFlow d="M257 247 H243 C225 247 225 222 203 222 H178 C151 222 155 182 126 182 H102 C80 182 83 126 56 126 H42" />
        </g>
      )}
      <text x="420" y="391" textAnchor="middle" className="method-svg-heading">
        SCALAR MLP
      </text>
      <text x="80" y="383" className="method-svg-note">
        Architecture schematic
      </text>
      <text x="80" y="399" className="method-svg-note">
        Signals illustrate data flow
      </text>
    </svg>
  );
}

function Track() {
  const [west, south, east, north] = example.map_bounds;
  const project = (lat: number, lon: number) => [
    ((lon - west) / (east - west)) * 280,
    ((north - lat) / (north - south)) * 180,
  ];
  const points = example.track.map((p) => project(p.lat, p.lon));
  const progress = 1;
  const index = progress * (points.length - 1);
  const a = points[Math.floor(index)],
    b = points[Math.min(points.length - 1, Math.floor(index) + 1)];
  const point = a.map((v, i) => v + (b[i] - v) * (index % 1));
  const [left, bottom, right, top] = example.grid.bounds;
  const crop = project(top, left);
  const cropEnd = project(bottom, right);
  const path = "M" + points.map((p) => p.join(",")).join("L");
  return (
    <div className="method-track">
      <svg
        viewBox="0 0 280 180"
        role="img"
        aria-label="Ian’s recorded track during the 72 hours before the selected scan, ending at the crop center"
      >
        <defs>
          <pattern
            id="method-map-grid"
            width="40"
            height="30"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M40 0H0V30"
              fill="none"
              stroke="#34485b"
              strokeWidth=".5"
            />
          </pattern>
        </defs>
        <rect width="280" height="180" fill="url(#method-map-grid)" />
        <image href="/method/coast.svg" width="280" height="180" />
        {(
          [
            [25.8, -81.5, "FLORIDA"],
            [21.8, -81, "CUBA"],
            [16.5, -79, "CARIBBEAN SEA"],
          ] as const
        ).map(([lat, lon, name]) => {
          const [x, y] = project(lat, lon);
          return (
            <text key={name} x={x} y={y}>
              {name}
            </text>
          );
        })}
        <path
          d={path}
          fill="none"
          stroke="#40c7bd"
          strokeOpacity=".2"
          strokeWidth="1.5"
        />
        <path
          d={path}
          pathLength="1"
          fill="none"
          stroke="#71ded1"
          strokeWidth="2"
          strokeDasharray="1"
          strokeDashoffset={1 - progress}
        />
        {points
          .filter((_, i) => i % 4 === 0)
          .map(([x, y], i) => (
            <circle key={i} cx={x} cy={y} r="1.7" fill="#7faaa9" />
          ))}
        <rect
          x={crop[0]}
          y={crop[1]}
          width={cropEnd[0] - crop[0]}
          height={cropEnd[1] - crop[1]}
          rx="2"
          fill="#40c7bd"
          fillOpacity=".05"
          stroke="#40c7bd"
          strokeOpacity={1}
          strokeDasharray="3 3"
        />
        <circle
          cx={point[0]}
          cy={point[1]}
          r="9"
          fill="#40c7bd"
          fillOpacity=".13"
        />
        <circle cx={point[0]} cy={point[1]} r="3.5" fill="#9af1db" />
      </svg>
      <div>
        <span>72-hour track</span>
        <span>Position → crop</span>
      </div>
    </div>
  );
}

export default function Method({ unit }: { unit: "kt" | "m/s" }) {
  const { container, reduced, paused, setPaused, motion } = useFlowMotion();
  const [mode, setMode] = useState<MethodMode>("joint");
  const [view, setView] = useState<"inference" | "training">("inference");
  const output = example.models[mode].scalars;
  return (
    <div className="method-page">
      <header className="method-heading">
        <div>
          <div className="eyebrow">
            <span className="tiny-line" /> Inside StormSense
          </div>
          <h1>
            From satellite imagery
            <br />
            to <em>surface winds.</em>
          </h1>
          <p>
            {view === "inference"
              ? "One observation. Two views of the storm."
              : "Two architectures. Learning from observed winds."}
          </p>
        </div>
        <div className="method-example-label">
          <span className="method-live-dot" />
          {view === "inference" ? (
            <>
              A REAL WORKED EXAMPLE
              <strong>Hurricane {example.storm.name}</strong>
              <span>{stamp(example.time)}</span>
              <small>GOES ABI · North Atlantic</small>
            </>
          ) : (
            <>
              SUPERVISED LEARNING
              <strong>Observed winds as targets</strong>
              <span>
                {mode === "joint"
                  ? "SAR + IBTrACS best track"
                  : "ATCF best-track fine-tuning"}
              </span>
              <small>Objectives for the example checkpoints</small>
            </>
          )}
        </div>
      </header>
      <section
        ref={container}
        className={`method-theater mode-${mode}`}
        aria-label="Interactive model walkthrough"
        data-motion={motion}
      >
        <div className="method-theater-top">
          <div
            className="method-view-tabs"
            role="tablist"
            aria-label="Method view"
          >
            {(["inference", "training"] as const).map((item, index) => (
              <button
                key={item}
                id={`method-tab-${item}`}
                role="tab"
                aria-selected={view === item}
                aria-controls={`method-panel-${item}`}
                tabIndex={view === item ? 0 : -1}
                onClick={() => setView(item)}
                onKeyDown={(event) => {
                  if (
                    ["ArrowLeft", "ArrowRight", "Home", "End"].includes(
                      event.key,
                    )
                  ) {
                    event.preventDefault();
                    const next =
                      event.key === "Home"
                        ? "inference"
                        : event.key === "End"
                          ? "training"
                          : index === 0
                            ? "training"
                            : "inference";
                    setView(next);
                    document.getElementById(`method-tab-${next}`)?.focus();
                  }
                }}
              >
                {item === "inference" ? "Inference" : "Training"}
              </button>
            ))}
          </div>
          <div className="method-theater-actions">
            <div
              className="method-mode"
              role="group"
              aria-label="Model architecture"
            >
              <button
                aria-pressed={mode === "joint"}
                onClick={() => setMode("joint")}
              >
                Joint field + scalars
              </button>
              <button
                aria-pressed={mode === "encoder"}
                onClick={() => setMode("encoder")}
              >
                Encoder-only scalars
              </button>
            </div>
            {!reduced && (
              <button
                className="method-motion-toggle"
                aria-label={paused ? "Resume flow" : "Pause flow"}
                title={paused ? "Resume flow" : "Pause flow"}
                onClick={() => setPaused((value) => !value)}
              >
                {paused ? "▶" : "Ⅱ"}
              </button>
            )}
          </div>
        </div>
        <div
          id="method-panel-inference"
          role="tabpanel"
          aria-labelledby="method-tab-inference"
          hidden={view !== "inference"}
        >
          <div className="method-scene">
            <div className="method-input">
              <div className="method-part-label">
                <span>01</span> THE INPUTS
              </div>
              <div className="method-input-stacks">
                <ImageStack
                  label="Satellite bands"
                  initial={6}
                  layers={example.channels.map((c) => ({
                    ...c,
                    label: c.id,
                    short: c.id.slice(-2),
                  }))}
                />
                <ImageStack
                  label="Extra tensors"
                  context
                  layers={example.context_channels.map((c) => ({
                    ...c,
                    short: c.short_label,
                  }))}
                />
              </div>
              <Track />
            </div>
            <div className="method-model">
              <div className="method-part-label">
                <span>02</span> THE MODEL{" "}
                <span className="method-model-kind">
                  {mode === "joint" ? "U-NET + MLP" : "ENCODER + MLP"}
                </span>
              </div>
              <Network mode={mode} />
              <div className="method-model-foot">
                <span className="method-small-dot" /> 10 satellite + 4 context
                channels + validity mask
              </div>
            </div>
            <div className="method-outputs">
              <div className="method-part-label">
                <span>03</span> THE PREDICTIONS
              </div>
              <div
                className={`method-field ${mode === "encoder" ? "is-omitted" : ""}`}
              >
                {mode === "joint" ? (
                  <>
                    <div className="method-field-image">
                      <img
                        src={example.field.image}
                        alt="Actual dense surface wind-speed prediction for Ian from the joint model"
                      />
                      <div className="method-field-grid" />
                      <span className="method-image-tag">DENSE WIND FIELD</span>
                    </div>
                    <div
                      className="method-colorbar"
                      style={{
                        background: `linear-gradient(to right, ${example.field.palette.join(",")})`,
                      }}
                    />
                    <div className="method-scale">
                      <span>0</span>
                      <span>Wind speed · {unit}</span>
                      <span>{wind(example.field.max_ms, unit)}</span>
                    </div>
                  </>
                ) : (
                  <div className="method-no-field">
                    <span aria-hidden="true">↳</span>
                    <h3>Scalars, directly.</h3>
                    <p>This variant omits the field decoder.</p>
                  </div>
                )}
              </div>
              <div
                className="method-scalars"
                aria-label={`${mode === "joint" ? "Joint model" : "Encoder-only model"} scalar predictions`}
              >
                <div className="method-scalar-title">
                  MAXIMUM SUSTAINED WIND <span>Vmax</span>
                </div>
                <div className="method-vmax">
                  <strong data-testid="method-vmax">
                    {wind(output.vmax_ms, unit)}
                  </strong>
                  <span>{unit}</span>
                  <small>Direct MLP estimate</small>
                </div>
                <div className="method-radii">
                  {(
                    [
                      ["RMW", output.rmw_km],
                      ["R34", output.r34_km],
                      ["R50", output.r50_km],
                      ["R64", output.r64_km],
                    ] as const
                  ).map(([name, value]) => (
                    <div key={name}>
                      <span>{name}</span>
                      <strong>
                        {value.toFixed(0)}
                        <small> km</small>
                      </strong>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
          <p className="method-accessible-summary">
            Satellite channels and storm-center/time context feed the encoder.
            Shared features branch into a wind-field decoder and a pooled scalar
            MLP. The encoder-only variant retains the scalar branch. Real inputs
            and predictions are fixed; animated signals illustrate data flow.
          </p>
        </div>
        <div
          id="method-panel-training"
          role="tabpanel"
          aria-labelledby="method-tab-training"
          hidden={view !== "training"}
        >
          {view === "training" && (
            <MethodTraining
              mode={mode}
              network={<Network mode={mode} training />}
              example={example}
              unit={unit}
            />
          )}
        </div>
      </section>
    </div>
  );
}

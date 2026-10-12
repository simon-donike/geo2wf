import { useEffect, useRef, type RefObject } from "react";
import { useLocation } from "react-router-dom";

type Box = { x: number; y: number; width: number; height: number };
const bounds = (target: HTMLElement): Box => {
  const rect = target.getBoundingClientRect();
  return { x: rect.left - 8, y: rect.top - 8, width: rect.width + 16, height: rect.height + 16 };
};

/** One persistent frame moves between sections, including across route changes. */
export default function ShowcaseHighlight({ anchor, entry, page, heading }: {
  anchor: string;
  entry: string;
  page: "overview" | "storm" | "method";
  heading: RefObject<HTMLHeadingElement | null>;
}) {
  const location = useLocation();
  const layer = useRef<HTMLDivElement>(null);
  const frame = useRef<HTMLDivElement>(null);
  const current = useRef<Box | null>(null);

  useEffect(() => {
    const content = document.getElementById("showcase-content")!;
    const card = document.querySelector<HTMLElement>(".showcase-controls")!;
    const reduced = matchMedia("(prefers-reduced-motion: reduce)");
    let target: HTMLElement | null = null;
    let animation = 0;
    let moving = false;
    let entered = false;
    const placeCard = (box?: Box) => {
      const width = card.offsetWidth, height = card.offsetHeight;
      const margin = innerWidth <= 800 ? 16 : 24;
      let x = innerWidth - width - margin;
      const y = innerHeight - height - margin;
      // Prefer the side away from the focused region; keep the card in view.
      if (box && box.x + box.width > x - 16 && box.x >= width + margin + 16)
        x = margin;
      card.style.left = `${Math.max(margin, x)}px`;
      card.style.top = `${Math.max(margin, y)}px`;
      card.style.right = "auto";
      card.style.bottom = "auto";
      content.style.setProperty("--tour-card-space", `${height}px`);
      return { x, y, width, height };
    };
    const draw = (box: Box) => {
      const viewport = content.getBoundingClientRect();
      layer.current!.style.top = `${viewport.top}px`;
      const cardBox = card.getBoundingClientRect();
      const overlapsX = box.x < cardBox.right + 12 && box.x + box.width > cardBox.left - 12;
      layer.current!.style.height = `${Math.max(0, (overlapsX ? Math.min(viewport.bottom, cardBox.top - 12) : viewport.bottom) - viewport.top)}px`;
      frame.current!.style.transform = `translate(${box.x}px, ${box.y - viewport.top}px)`;
      frame.current!.style.width = `${box.width}px`;
      frame.current!.style.height = `${box.height}px`;
      layer.current!.style.opacity = "1";
      current.current = box;
    };
    const visible = (element: HTMLElement | null): element is HTMLElement =>
      !!element?.isConnected && !!element.getClientRects().length && !element.closest("[hidden]");
    const follow = () => {
      if (moving) return;
      if (visible(target)) { placeCard(bounds(target)); draw(bounds(target)); }
      else layer.current!.style.opacity = "0";
    };
    const cancel = () => {
      cancelAnimationFrame(animation);
      moving = false;
      frame.current!.dataset.moving = "false";
      follow();
    };
    const resize = new ResizeObserver(follow);
    resize.observe(content);
    resize.observe(card);
    const find = () => {
      const onPage = page === "overview" ? location.pathname === "/" : page === "method" ? location.pathname === "/method" : location.pathname === "/storms/EP152026";
      const next = onPage ? content.querySelector<HTMLElement>(`[data-tour="${anchor}"]`) : null;
      if (!visible(next)) {
        target?.removeAttribute("data-tour-highlight");
        target = null;
        cancel();
        return;
      }
      if (next === target) return;
      if (target) {
        target.removeAttribute("data-tour-highlight");
        resize.unobserve(target);
      }
      target = next;
      target.setAttribute("data-tour-highlight", "true");
      resize.observe(target);
      if (entered) { follow(); return; }
      entered = true;
      heading.current?.focus({ preventScroll: true });
      const viewport = content.getBoundingClientRect();
      const rect = target.getBoundingClientRect();
      const placement = placeCard(bounds(target));
      const overlapsX = rect.left < placement.x + placement.width + 16 && rect.right > placement.x - 16;
      const visibleBottom = overlapsX ? placement.y - 16 : viewport.bottom - 16;
      const startScroll = content.scrollTop;
      // Keep the viewport still when the next section is already fully visible.
      const fits = rect.top >= viewport.top + 16 && rect.bottom <= visibleBottom;
      const endScroll = fits ? startScroll : Math.max(0, Math.min(
        content.scrollHeight - content.clientHeight,
        startScroll + rect.top - viewport.top - 20,
      ));
      const from = current.current ?? bounds(target);
      const start = performance.now();
      moving = true;
      frame.current!.dataset.moving = "true";
      const tick = (now: number) => {
        if (!visible(target)) { cancel(); return; }
        const progress = reduced.matches ? 1 : Math.min(1, (now - start) / 480);
        const ease = 1 - Math.pow(1 - progress, 3);
        content.scrollTop = startScroll + (endScroll - startScroll) * ease;
        const to = bounds(target);
        draw({
          x: from.x + (to.x - from.x) * ease,
          y: from.y + (to.y - from.y) * ease,
          width: from.width + (to.width - from.width) * ease,
          height: from.height + (to.height - from.height) * ease,
        });
        if (progress < 1) animation = requestAnimationFrame(tick);
        else { moving = false; frame.current!.dataset.moving = "false"; }
      };
      animation = requestAnimationFrame(tick);
    };
    const observer = new MutationObserver(find);
    observer.observe(content, { childList: true, subtree: true, attributes: true, attributeFilter: ["hidden"] });
    content.addEventListener("scroll", follow, { passive: true });
    content.addEventListener("wheel", cancel, { passive: true });
    content.addEventListener("pointerdown", cancel);
    content.addEventListener("keydown", cancel);
    window.addEventListener("resize", follow);
    reduced.addEventListener("change", cancel);
    placeCard();
    find();
    return () => {
      cancelAnimationFrame(animation);
      observer.disconnect();
      resize.disconnect();
      target?.removeAttribute("data-tour-highlight");
      content.removeEventListener("scroll", follow);
      content.removeEventListener("wheel", cancel);
      content.removeEventListener("pointerdown", cancel);
      content.removeEventListener("keydown", cancel);
      window.removeEventListener("resize", follow);
      reduced.removeEventListener("change", cancel);
    };
  }, [anchor, entry, page, location.pathname, heading]);

  return <div ref={layer} className="showcase-highlight-layer" aria-hidden="true" style={{ opacity: 0 }}>
    <div ref={frame} className="showcase-highlight" />
  </div>;
}

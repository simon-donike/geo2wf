import "leaflet";

// Leaflet 1.9.4 ImageOverlay._initImage explicitly accepts an existing IMG.
// The factory's DefinitelyTyped signature omits this supported overload.
declare module "leaflet" {
  export function imageOverlay(
    image: HTMLImageElement,
    bounds: LatLngBoundsExpression,
    options?: ImageOverlayOptions,
  ): ImageOverlay;
}

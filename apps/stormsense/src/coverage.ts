/** NOAA National Hurricane Operations Plan (2023), section 2.2.1 / Appendix O.
 * https://www.weather.gov/media/nws/IHC2023/2023_nhop.pdf#page=192
 * Figure O-1's ocean basin limits, simplified to its straight meridians/equator.
 * Coastlines complete the ocean boundaries; this is not a satellite footprint
 * or a guarantee that model estimates exist everywhere within these basins.
 * Latitude endpoints on the two meridians meet land as in NOAA's schematic;
 * they do not impose a northern latitude cutoff on the storm feed.
 */
export const COVERAGE_SOURCE =
  "https://www.weather.gov/media/nws/IHC2023/2023_nhop.pdf#page=192";
export const COVERAGE_LIMITS: { name: string; points: [number, number][] }[] = [
  { name: "Central Pacific western limit · 180°", points: [[0, -180], [65, -180]] },
  { name: "Pacific southern limit · Equator", points: [[0, -180], [0, -80.1]] },
  { name: "Atlantic southern limit · Equator", points: [[0, -50], [0, 9.3]] },
];

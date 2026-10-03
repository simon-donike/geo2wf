/** A bounded, cancellable decode cache. Selected images always take priority.
 * Compressed browser HTTP caching is separate from this small decoded window.
 */
export interface DecodedImage {
  image: HTMLImageElement;
  dispose(): void;
}
type State = "loading" | "ready" | "error";
interface Entry {
  state: State;
  controller: AbortController;
  decoded?: DecodedImage;
  bytes: number;
}
export type ImageLoader = (
  url: string,
  signal: AbortSignal,
) => Promise<DecodedImage>;
export async function decodeBlob(
  blob: Blob,
  signal: AbortSignal,
): Promise<DecodedImage> {
  signal.throwIfAborted();
  const objectURL = URL.createObjectURL(blob);
  const image = new Image();
  image.decoding = "async";
  image.src = objectURL;
  try {
    await image.decode();
    signal.throwIfAborted();
    return {
      image,
      dispose: () => {
        image.src = "";
        URL.revokeObjectURL(objectURL);
      },
    };
  } catch (error) {
    URL.revokeObjectURL(objectURL);
    throw error;
  }
}
export const decodeImage: ImageLoader = async (url, signal) => {
  const response = await fetch(url, {
    signal: AbortSignal.any([signal, AbortSignal.timeout(15000)]),
  });
  if (!response.ok) throw Error(`Image unavailable (${response.status})`);
  return decodeBlob(await response.blob(), signal);
};

export class ImageCache {
  private entries = new Map<string, Entry>();
  private wanted: string[] = [];
  private active = 0;
  private closed = false;
  constructor(
    private notify: () => void,
    private loader: ImageLoader = decodeImage,
    private budget = 48 * 1024 * 1024,
    private concurrency = 4,
  ) {}
  get(url: string) {
    return this.entries.get(url);
  }
  get decodedBytes() {
    return [...this.entries.values()].reduce((n, e) => n + e.bytes, 0);
  }
  get size() {
    return this.entries.size;
  }
  window(urls: string[]) {
    this.wanted = [...new Set(urls)];
    const wanted = new Set(this.wanted);
    for (const [url, entry] of this.entries) {
      if (!wanted.has(url)) this.remove(url, entry);
    }
    this.pump();
  }
  private remove(url: string, entry: Entry) {
    this.entries.delete(url);
    entry.controller.abort();
    entry.decoded?.dispose();
  }
  private pump() {
    if (this.closed) return;
    for (const url of this.wanted) {
      if (this.active >= this.concurrency) break;
      if (this.entries.has(url)) continue;
      // Estimates are conservative for 768px full images; stop background
      // decoding before the budget, while always allowing the selected parts.
      if (this.decodedBytes > this.budget - this.concurrency * 768 * 768 * 4)
        break;
      const entry: Entry = {
        state: "loading",
        controller: new AbortController(),
        bytes: 0,
      };
      this.entries.set(url, entry);
      this.active++;
      void this.loader(url, entry.controller.signal)
        .then((decoded) => {
          if (this.closed || this.entries.get(url) !== entry) {
            decoded.dispose();
            return;
          }
          entry.decoded = decoded;
          entry.bytes =
            decoded.image.naturalWidth * decoded.image.naturalHeight * 4;
          entry.state = "ready";
        })
        .catch(() => {
          if (this.entries.get(url) === entry) entry.state = "error";
        })
        .finally(() => {
          this.active--;
          if (!this.closed) {
            this.notify();
            this.pump();
          }
        });
    }
  }
  retry() {
    for (const [url, entry] of this.entries)
      if (entry.state === "error") this.remove(url, entry);
    this.pump();
  }
  close() {
    this.closed = true;
    for (const [url, entry] of this.entries) this.remove(url, entry);
  }
}

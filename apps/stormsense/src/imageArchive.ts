/** Download a whole storm once, retaining compressed pixels separately from
 * the small decoded-image window. Scrubbing never cancels useful downloads.
 */
import { unzipSync } from "fflate";
import { dataUrl } from "./data";
import type { ImageBundle } from "./types";
type DownloadState = "queued" | "loading" | "ready" | "error";
type BlobLoader = (url: string, signal: AbortSignal) => Promise<Blob>;
interface Download {
  state: DownloadState;
  controller: AbortController;
  promise: Promise<Blob>;
  resolve: (blob: Blob) => void;
  reject: (error: unknown) => void;
  bytes: number;
}
const fetchBlob: BlobLoader = async (url, signal) => {
  const response = await fetch(url, {
    signal: AbortSignal.any([
      signal,
      AbortSignal.timeout(url.endsWith(".zip") ? 60000 : 15000),
    ]),
  });
  if (!response.ok) throw Error(`Image unavailable (${response.status})`);
  return response.blob();
};

export class ImageArchive {
  private entries = new Map<string, Download>();
  private order: string[] = [];
  private active = 0;
  private paused = false;
  private closed = false;
  private bundles = new Map<string, ImageBundle>();
  private bundleRequests = new Map<
    string,
    {
      promise: Promise<Map<string, Blob>>;
      controller: AbortController;
      failed: boolean;
      images: string[];
    }
  >();
  private bundleControllers = new Set<AbortController>();
  constructor(
    private notify: () => void,
    private loader: BlobLoader = fetchBlob,
    private concurrency = 8,
  ) {}
  state(url: string) {
    return this.entries.get(url)?.state;
  }
  get compressedBytes() {
    return [...this.entries.values()].reduce((n, entry) => n + entry.bytes, 0);
  }
  private entry(): Download {
    let resolve!: Download["resolve"], reject!: Download["reject"];
    const promise = new Promise<Blob>((yes, no) => {
      resolve = yes;
      reject = no;
    });
    // Background downloads may fail before anyone selects their hour.
    void promise.catch(() => {});
    return {
      state: "queued",
      controller: new AbortController(),
      promise,
      resolve,
      reject,
      bytes: 0,
    };
  }
  set(
    urls: string[],
    priority: string[],
    paused = false,
    bundles: ImageBundle[] = [],
  ) {
    if (this.closed) return;
    this.paused = paused;
    const retained = new Set(urls);
    this.bundles = new Map(
      bundles.flatMap((bundle) =>
        bundle.images.map((path) => [dataUrl(path), bundle] as const),
      ),
    );
    const bundlePaths = new Set(bundles.map((bundle) => bundle.path));
    for (const [path, request] of this.bundleRequests) {
      if (!bundlePaths.has(path)) {
        this.bundleRequests.delete(path);
        // A refreshed current-day pack may retain images already being read
        // from its predecessor. Let those consumers finish before releasing it.
        if (!request.images.some((url) => retained.has(url)))
          request.controller.abort();
      }
    }
    for (const [url, entry] of this.entries) {
      if (!retained.has(url)) {
        this.entries.delete(url);
        entry.controller.abort();
        entry.reject(new DOMException("Storm changed", "AbortError"));
      }
    }
    for (const url of retained)
      if (!this.entries.has(url)) this.entries.set(url, this.entry());
    this.order = [
      ...new Set([...priority.filter((url) => retained.has(url)), ...urls]),
    ];
    this.pump();
  }
  private load(url: string, signal: AbortSignal): Promise<Blob> {
    const bundle = this.bundles.get(url);
    if (!bundle) return this.loader(url, signal); // Earlier release compatibility.
    let request = this.bundleRequests.get(bundle.path);
    if (!request) {
      const controller = new AbortController();
      this.bundleControllers.add(controller);
      const entry = {
        controller,
        failed: false,
        images: bundle.images.map(dataUrl),
        promise: Promise.resolve(new Map<string, Blob>()),
      };
      entry.promise = this.loader(dataUrl(bundle.path), controller.signal)
        .then(async (blob) => {
          if (blob.size !== bundle.bytes || blob.size > 8 * 1024 * 1024)
            throw Error("Invalid image bundle size");
          const buffer = await blob.arrayBuffer();
          const hash = [
            ...new Uint8Array(await crypto.subtle.digest("SHA-256", buffer)),
          ]
            .map((b) => b.toString(16).padStart(2, "0"))
            .join("");
          if (hash !== bundle.sha256)
            throw Error("Image bundle checksum mismatch");
          controller.signal.throwIfAborted();
          const wanted = new Set(bundle.images);
          const files = unzipSync(new Uint8Array(buffer), {
            filter: (file) =>
              wanted.has(file.name) && file.originalSize <= 1024 * 1024,
          });
          const images = new Map<string, Blob>();
          for (const path of wanted) {
            if (!files[path]) throw Error("Image missing from daily bundle");
            images.set(
              dataUrl(path),
              new Blob([new Uint8Array(files[path])], { type: "image/webp" }),
            );
          }
          return images;
        })
        .catch((error) => {
          entry.failed = true;
          throw error;
        })
        .finally(() => this.bundleControllers.delete(controller));
      request = entry;
      this.bundleRequests.set(bundle.path, entry);
    }
    return request.promise.then((files) => {
      signal.throwIfAborted();
      const image = files.get(url);
      if (!image) throw Error("Image missing from daily bundle");
      return image;
    });
  }
  /** Cancel just the consumer's wait, preserving the shared background fetch. */
  read(url: string, signal: AbortSignal): Promise<Blob> {
    if (signal.aborted) return Promise.reject(signal.reason);
    const entry = this.entries.get(url);
    if (!entry)
      return Promise.reject(Error("Image is outside the current storm"));
    return new Promise((resolve, reject) => {
      const abort = () => {
        signal.removeEventListener("abort", abort);
        reject(signal.reason);
      };
      signal.addEventListener("abort", abort, { once: true });
      void entry.promise.then(
        (blob) => {
          signal.removeEventListener("abort", abort);
          resolve(blob);
        },
        (error) => {
          signal.removeEventListener("abort", abort);
          reject(error);
        },
      );
    });
  }
  private pump() {
    if (this.closed || this.paused) return;
    for (const url of this.order) {
      if (this.active >= this.concurrency) break;
      const entry = this.entries.get(url)!;
      if (entry.state !== "queued") continue;
      entry.state = "loading";
      this.active++;
      void this.load(url, entry.controller.signal)
        .then((blob) => {
          if (this.closed || this.entries.get(url) !== entry) return;
          entry.bytes = blob.size;
          entry.state = "ready";
          entry.resolve(blob);
        })
        .catch((error) => {
          entry.state = "error";
          entry.reject(error);
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
    for (const [path, request] of this.bundleRequests)
      if (request.failed) this.bundleRequests.delete(path);
    for (const [url, entry] of this.entries)
      if (entry.state === "error") this.entries.set(url, this.entry());
    this.pump();
  }
  close() {
    this.closed = true;
    for (const controller of this.bundleControllers) controller.abort();
    this.bundleRequests.clear();
    this.bundleControllers.clear();
    this.bundles.clear();
    for (const entry of this.entries.values()) {
      entry.controller.abort();
      entry.reject(new DOMException("Storm closed", "AbortError"));
    }
    this.entries.clear();
    this.order = [];
  }
}

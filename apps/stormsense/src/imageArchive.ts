/** Download a whole storm once, retaining compressed pixels separately from
 * the small decoded-image window. Scrubbing never cancels useful downloads.
 */
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
    signal: AbortSignal.any([signal, AbortSignal.timeout(15000)]),
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
  set(urls: string[], priority: string[], paused = false) {
    if (this.closed) return;
    this.paused = paused;
    const retained = new Set(urls);
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
      void this.loader(url, entry.controller.signal)
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
    for (const [url, entry] of this.entries)
      if (entry.state === "error") this.entries.set(url, this.entry());
    this.pump();
  }
  close() {
    this.closed = true;
    for (const entry of this.entries.values()) {
      entry.controller.abort();
      entry.reject(new DOMException("Storm closed", "AbortError"));
    }
    this.entries.clear();
    this.order = [];
  }
}

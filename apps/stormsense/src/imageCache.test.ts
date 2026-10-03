import { describe, it, expect, vi } from "vitest";
import { ImageCache, type DecodedImage, type ImageLoader } from "./imageCache";

function fixture() {
  const pending = new Map<
    string,
    {
      signal: AbortSignal;
      resolve: (value: DecodedImage) => void;
      reject: (error: Error) => void;
    }
  >();
  const loader: ImageLoader = (url, signal) =>
    new Promise((resolve, reject) => {
      pending.set(url, { signal, resolve, reject });
      signal.addEventListener("abort", () => reject(Error("aborted")));
    });
  return { pending, loader };
}
const decoded = (dispose = vi.fn()): DecodedImage => ({
  image: { naturalWidth: 768, naturalHeight: 768 } as HTMLImageElement,
  dispose,
});
const flush = async () => {
  for (let i = 0; i < 8; i++) await Promise.resolve();
};

describe("bounded imagery buffer", () => {
  it("prioritizes a jump and cancels obsolete fetches", async () => {
    const { loader, pending } = fixture();
    const cache = new ImageCache(vi.fn(), loader, 48 * 1024 * 1024, 2);
    cache.window(["selected", "next", "later"]);
    expect([...pending.keys()]).toEqual(["selected", "next"]);
    cache.window(["jump", "jump-next"]);
    expect(pending.get("selected")!.signal.aborted).toBe(true);
    await flush();
    expect(pending.has("jump")).toBe(true);
    expect(cache.get("selected")).toBeUndefined();
    pending.get("jump")!.resolve(decoded());
    await flush();
    expect(cache.get("jump")?.state).toBe("ready");
    cache.close();
  });
  it("reuses decoded frames and frees pixels when outside the window", async () => {
    const { loader, pending } = fixture();
    const notify = vi.fn();
    const dispose = vi.fn();
    const cache = new ImageCache(notify, loader);
    cache.window(["a"]);
    pending.get("a")!.resolve(decoded(dispose));
    await flush();
    expect(cache.decodedBytes).toBe(768 * 768 * 4);
    const image = cache.get("a");
    cache.window(["a"]);
    expect(cache.get("a")).toBe(image);
    cache.window(["b"]);
    expect(dispose).toHaveBeenCalledOnce();
    expect(cache.decodedBytes).toBe(0);
    cache.close();
    const count = notify.mock.calls.length;
    await flush();
    expect(notify.mock.calls.length).toBe(count);
  });
  it("bounds concurrency and decoded memory across long histories", async () => {
    const notify = vi.fn();
    const loader = vi.fn(async () => decoded());
    const cache = new ImageCache(notify, loader, 24 * 1024 * 1024, 2);
    cache.window(Array.from({ length: 500 }, (_, i) => String(i)));
    for (let i = 0; i < 20; i++) await flush();
    expect(cache.decodedBytes).toBeLessThanOrEqual(24 * 1024 * 1024);
    expect(cache.size).toBeLessThan(12);
    cache.close();
  });
});

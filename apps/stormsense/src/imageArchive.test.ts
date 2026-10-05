import { describe, it, expect, vi } from "vitest";
import { ImageArchive } from "./imageArchive";
import { zipSync } from "fflate";
import { dataUrl } from "./data";
import type { ImageBundle } from "./types";

async function packed() {
  const files = {
    "imagery/a.webp": new Uint8Array([1, 2, 3]),
    "imagery/b.webp": new Uint8Array([4, 5]),
    "imagery/a.webp.aux.xml": new Uint8Array([6]),
  };
  const zip = new Uint8Array(zipSync(files, { level: 0 }));
  const hash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", zip))]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
  const bundle: ImageBundle = {
    schema_version: 1,
    date: "2026-09-01",
    path: `bundles/${hash}.zip`,
    sha256: hash,
    bytes: zip.length,
    images: ["imagery/a.webp", "imagery/b.webp"],
  };
  return { bundle, blob: new Blob([zip]), urls: bundle.images.map(dataUrl) };
}

const flush = async () => {
  for (let i = 0; i < 8; i++) await Promise.resolve();
};
function fixture() {
  const pending = new Map<
    string,
    {
      signal: AbortSignal;
      resolve: (blob: Blob) => void;
      reject: (error: Error) => void;
    }
  >();
  const loader = vi.fn(
    (url: string, signal: AbortSignal) =>
      new Promise<Blob>((resolve, reject) => {
        pending.set(url, { signal, resolve, reject });
        signal.addEventListener("abort", () => reject(Error("aborted")));
      }),
  );
  return { pending, loader };
}
describe("whole-storm compressed archive", () => {
  it("fetches a daily bundle once, validates it and serves all images locally", async () => {
    const { bundle, blob, urls } = await packed();
    const loader = vi.fn(async (_url: string, _signal: AbortSignal) => blob);
    const archive = new ImageArchive(vi.fn(), loader);
    archive.set(urls, [urls[1]], false, [bundle]);
    const first = await archive.read(urls[0], new AbortController().signal);
    const second = await archive.read(urls[1], new AbortController().signal);
    expect([...new Uint8Array(await first.arrayBuffer())]).toEqual([1, 2, 3]);
    expect(second.size).toBe(2);
    archive.set([...urls], [], false, [{ ...bundle }]);
    expect(loader).toHaveBeenCalledOnce();
    expect(loader.mock.calls[0][0]).toBe(dataUrl(bundle.path));
    archive.close();
  });
  it("rejects corrupted packs and retries only the pack, without individual requests", async () => {
    const { bundle, blob, urls } = await packed();
    const loader = vi
      .fn()
      .mockResolvedValueOnce(new Blob([new Uint8Array(blob.size)]))
      .mockResolvedValue(blob);
    const archive = new ImageArchive(vi.fn(), loader);
    archive.set(urls, [], false, [bundle]);
    await expect(
      archive.read(urls[0], new AbortController().signal),
    ).rejects.toThrow("checksum");
    await vi.waitFor(() => expect(archive.state(urls[1])).toBe("error"));
    archive.retry();
    await expect(
      archive.read(urls[1], new AbortController().signal),
    ).resolves.toHaveProperty("size", 2);
    expect(loader).toHaveBeenCalledTimes(2);
    expect(loader.mock.calls.every(([url]) => url.endsWith(".zip"))).toBe(true);
    archive.close();
  });
  it("downloads every image with bounded concurrency and selected-hour priority", async () => {
    const { pending, loader } = fixture();
    const archive = new ImageArchive(vi.fn(), loader, 2);
    archive.set(["a", "b", "c", "d"], ["c"]);
    expect([...pending.keys()]).toEqual(["c", "a"]);
    archive.set(["a", "b", "c", "d"], ["d"]);
    pending.get("a")!.resolve(new Blob(["aa"]));
    await flush();
    expect(pending.has("d")).toBe(true);
    expect(pending.has("b")).toBe(false);
    pending.get("c")!.resolve(new Blob(["ccc"]));
    pending.get("d")!.resolve(new Blob(["d"]));
    await flush();
    pending.get("b")!.resolve(new Blob(["bb"]));
    await flush();
    for (const url of ["a", "b", "c", "d"])
      expect(archive.state(url)).toBe("ready");
    expect(archive.compressedBytes).toBe(8);
    expect(
      await (await archive.read("a", new AbortController().signal)).text(),
    ).toBe("aa");
    expect(loader).toHaveBeenCalledTimes(4);
    archive.close();
  });
  it("cancels a stale decode wait immediately but preserves its useful download", async () => {
    const { pending, loader } = fixture();
    const archive = new ImageArchive(vi.fn(), loader);
    archive.set(["a"], ["a"]);
    const consumer = new AbortController();
    const read = archive.read("a", consumer.signal);
    const rejected = expect(read).rejects.toMatchObject({ name: "AbortError" });
    consumer.abort();
    await rejected;
    expect(pending.get("a")!.signal.aborted).toBe(false);
    pending.get("a")!.resolve(new Blob(["saved"]));
    await flush();
    expect(
      await (await archive.read("a", new AbortController().signal)).text(),
    ).toBe("saved");
    expect(loader).toHaveBeenCalledOnce();
    archive.close();
  });
  it("retries failures, retains downloads across refreshes and frees the prior storm", async () => {
    const { pending, loader } = fixture();
    const archive = new ImageArchive(vi.fn(), loader);
    archive.set(["a", "b"], []);
    pending.get("a")!.resolve(new Blob(["saved"]));
    pending.get("b")!.reject(Error("offline"));
    await flush();
    archive.set(["a", "b", "c"], ["b"]);
    archive.retry();
    pending.get("b")!.resolve(new Blob(["retried"]));
    await flush();
    expect(archive.state("b")).toBe("ready");
    expect(loader.mock.calls.filter(([url]) => url === "a")).toHaveLength(1);
    archive.set(["other-storm"], []);
    expect(pending.get("c")!.signal.aborted).toBe(true);
    expect(archive.state("a")).toBeUndefined();
    expect(archive.compressedBytes).toBe(0);
    archive.close();
    expect(pending.get("other-storm")!.signal.aborted).toBe(true);
  });
});

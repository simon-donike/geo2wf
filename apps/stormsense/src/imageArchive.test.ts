import { describe, it, expect, vi } from "vitest";
import { ImageArchive } from "./imageArchive";

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

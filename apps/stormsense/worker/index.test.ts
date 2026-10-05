import { afterEach, beforeEach, describe, it, expect, vi } from "vitest";
import worker from "./index";
let stored: Map<string, { body: ArrayBuffer; headers: Headers }>;
let cache: { match: ReturnType<typeof vi.fn>; put: ReturnType<typeof vi.fn> };
beforeEach(() => {
  stored = new Map();
  cache = {
    match: vi.fn(async (request: Request) => {
      const entry = stored.get(request.url);
      return entry
        ? new Response(entry.body, { headers: entry.headers })
        : undefined;
    }),
    put: vi.fn(async (request: Request, response: Response) => {
      stored.set(request.url, {
        body: await response.arrayBuffer(),
        headers: response.headers,
      });
    }),
  };
  vi.stubGlobal("caches", { default: cache });
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});
function env(fail = false) {
  const body = '{"schema_version":1}';
  return {
    DATA_PREFIX: "explorer/stormsense/",
    STORMSENSE_DATA: {
      get: vi.fn(async () => {
        if (fail) throw Error("offline");
        return { body, size: body.length, httpEtag: '"abc"' };
      }),
      head: vi.fn(async () => ({ size: body.length, httpEtag: '"abc"' })),
    },
    ASSETS: { fetch: vi.fn(async () => new Response("app")) },
  };
}
const read = async (
  path: string,
  e: ReturnType<typeof env>,
  options?: RequestInit,
) => {
  const background: Promise<unknown>[] = [];
  const response = await worker.fetch(
    new Request("https://test" + path, options) as Parameters<
      typeof worker.fetch
    >[0],
    e as never,
    {
      waitUntil: (promise: Promise<unknown>) => background.push(promise),
    } as never,
  );
  await Promise.all(background);
  return response;
};
describe("read-only data Worker", () => {
  it("restricts routes and methods", async () => {
    const e = env();
    expect((await read("/data/secret.json", e)).status).toBe(404);
    expect(
      (await read("/data/latest.json", e, { method: "POST" })).status,
    ).toBe(405);
    expect(e.STORMSENSE_DATA.get).not.toHaveBeenCalled();
  });
  it("uses isolated R2 keys and revalidates the pointer", async () => {
    const e = env();
    const response = await read("/data/latest.json", e);
    expect(e.STORMSENSE_DATA.get).toHaveBeenCalledWith(
      "explorer/stormsense/latest.json",
    );
    expect(response.headers.get("Cache-Control")).toBe("no-cache");
    expect(
      (
        await read("/data/latest.json", e, {
          headers: { "If-None-Match": '"abc"' },
        })
      ).status,
    ).toBe(304);
    expect(e.STORMSENSE_DATA.get).toHaveBeenCalledTimes(2);
    expect(cache.match).not.toHaveBeenCalled();
    expect(cache.put).not.toHaveBeenCalled();
    expect(response.headers.get("X-StormSense-Cache")).toBe("BYPASS");
  });
  it("caches immutable releases and supports HEAD", async () => {
    const response = await read(
      "/data/releases/20261002T000000Z/catalog.json",
      env(),
      { method: "HEAD" },
    );
    expect(response.headers.get("Cache-Control")).toContain("immutable");
    expect(await response.text()).toBe("");
  });
  it("revalidates edge-compressed, listed and wildcard ETags", async () => {
    for (const tag of ['W/"abc"', '"old", W/"abc"', "*"]) {
      const response = await read("/data/latest.json", env(), {
        headers: { "If-None-Match": tag },
      });
      expect(response.status).toBe(304);
      expect(await response.text()).toBe("");
    }
    expect(
      (
        await read("/data/latest.json", env(), {
          headers: { "If-None-Match": 'W/"different"' },
        })
      ).status,
    ).toBe(200);
  });
  it("reports R2 failure without inventing an empty catalog", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    expect((await read("/data/latest.json", env(true))).status).toBe(503);
    vi.restoreAllMocks();
  });
  it("serves immutable images and GIS metadata with correct MIME types", async () => {
    const hash = "a".repeat(64);
    for (const [suffix, type] of [
      ["webp", "image/webp"],
      ["webp.aux.xml", "application/xml"],
      ["json", "application/json"],
    ]) {
      const e = env();
      const response = await read(`/data/imagery/${hash}.${suffix}`, e);
      expect(response.status).toBe(200);
      expect(response.headers.get("Content-Type")).toBe(type);
      expect(response.headers.get("Cache-Control")).toContain("immutable");
      expect(response.headers.get("Access-Control-Allow-Origin")).toBe("*");
      expect(
        (await read(`/data/imagery/${hash}.${suffix}`, e, { method: "PUT" }))
          .status,
      ).toBe(405);
    }
    expect((await read(`/data/imagery/${hash}.html`, env())).status).toBe(404);
  });
  it("reuses an edge-cached image for GET, HEAD and conditional requests", async () => {
    const e = env();
    const path = `/data/imagery/${"b".repeat(64)}.webp`;
    const first = await read(path + "?first=1", e);
    expect(first.headers.get("X-StormSense-Cache")).toBe("MISS");
    const second = await read(path + "?different=2", e);
    expect(second.headers.get("X-StormSense-Cache")).toBe("HIT");
    expect(await second.text()).toBe(await first.text());
    const head = await read(path, e, { method: "HEAD" });
    expect(head.headers.get("Content-Length")).toBe("20");
    expect(await head.text()).toBe("");
    const unchanged = await read(path, e, {
      headers: { "If-None-Match": 'W/"abc"' },
    });
    expect(unchanged.status).toBe(304);
    expect(unchanged.headers.get("X-StormSense-Cache")).toBe("HIT");
    expect(await unchanged.text()).toBe("");
    expect(e.STORMSENSE_DATA.get).toHaveBeenCalledTimes(1);
    expect(e.STORMSENSE_DATA.head).not.toHaveBeenCalled();
  });
  it("never stores an empty HEAD response or a source failure", async () => {
    const path = `/data/objects/${"c".repeat(64)}.json`;
    const e = env();
    await read(path, e, { method: "HEAD" });
    expect(e.STORMSENSE_DATA.head).toHaveBeenCalledTimes(1);
    expect(e.STORMSENSE_DATA.get).not.toHaveBeenCalled();
    expect(cache.put).not.toHaveBeenCalled();
    vi.spyOn(console, "error").mockImplementation(() => {});
    expect((await read(path, env(true))).status).toBe(503);
    expect(cache.put).not.toHaveBeenCalled();
    e.STORMSENSE_DATA.get.mockResolvedValueOnce(null as never);
    expect((await read(path, e)).status).toBe(404);
    expect(cache.put).not.toHaveBeenCalled();
    const recovered = await read(path, e);
    expect(recovered.status).toBe(200);
    expect(await recovered.text()).toContain("schema_version");
  });
  it("still delivers R2 data when cache reads or writes fail", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    cache.match.mockRejectedValueOnce(Error("cache unavailable"));
    cache.put.mockRejectedValueOnce(Error("cache full"));
    const response = await read(`/data/objects/${"d".repeat(64)}.json`, env());
    expect(response.status).toBe(200);
    expect(await response.text()).toContain("schema_version");
  });
  it("does not tee an unbounded object into cache", async () => {
    const e = env();
    e.STORMSENSE_DATA.get.mockResolvedValueOnce({
      body: "large stream",
      size: 9 * 1024 * 1024,
      httpEtag: '"large"',
    });
    expect((await read(`/data/objects/${"e".repeat(64)}.json`, e)).status).toBe(
      200,
    );
    expect(cache.put).not.toHaveBeenCalled();
  });
  it("serves immutable daily ZIPs through the same edge cache", async () => {
    const e = env();
    const path = `/data/bundles/${"a".repeat(64)}.zip`;
    const first = await read(path, e);
    expect(first.headers.get("Content-Type")).toBe("application/zip");
    expect((await read(path, e)).headers.get("X-StormSense-Cache")).toBe("HIT");
    expect(e.STORMSENSE_DATA.get).toHaveBeenCalledOnce();
    expect((await read("/data/bundles/arbitrary.zip", e)).status).toBe(404);
  });
});

import { describe, it, expect, vi } from "vitest";
import worker from "./index";
function env(fail = false) {
  return {
    DATA_PREFIX: "explorer/stormsense/",
    STORMSENSE_DATA: {
      get: vi.fn(async () => {
        if (fail) throw Error("offline");
        return { body: '{"schema_version":1}', httpEtag: '"abc"' };
      }),
    },
    ASSETS: { fetch: vi.fn(async () => new Response("app")) },
  };
}
const read = (path: string, e: ReturnType<typeof env>, options?: RequestInit) =>
  worker.fetch(
    new Request("https://test" + path, options) as Parameters<
      typeof worker.fetch
    >[0],
    e as never,
  );
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
});

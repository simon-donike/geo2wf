function clientResponse(
  request: Request,
  response: Response,
  cacheStatus: string,
  ctx: ExecutionContext,
) {
  const headers = new Headers(response.headers);
  headers.set("X-StormSense-Cache", cacheStatus);
  // GET/HEAD use weak comparison; edge compression can add the W/ prefix.
  const notModified = request.headers
    .get("If-None-Match")
    ?.split(",")
    .some((value) => {
      const tag = value.trim();
      return tag === "*" || tag.replace(/^W\//, "") === headers.get("ETag");
    });
  if (notModified || request.method === "HEAD") {
    if (response.body) ctx.waitUntil(response.body.cancel().catch(() => {}));
    return new Response(null, { status: notModified ? 304 : 200, headers });
  }
  return new Response(response.body, { headers });
}

function cacheError(event: string, key: string, error: unknown) {
  console.error(JSON.stringify({ event, key, error: String(error) }));
}

export default {
  async fetch(request, env, ctx): Promise<Response> {
    const url = new URL(request.url);
    if (!url.pathname.startsWith("/data/")) return env.ASSETS.fetch(request);
    if (!["GET", "HEAD"].includes(request.method))
      return new Response("Method not allowed", {
        status: 405,
        headers: { Allow: "GET, HEAD" },
      });
    const key = url.pathname.slice("/data/".length);
    if (
      !/^(latest\.json|releases\/[A-Za-z0-9_-]+\/(catalog|coverage|evaluation)\.json|objects\/[a-f0-9]{64}\.json|imagery\/[a-f0-9]{64}\.(?:json|webp(?:\.aux\.xml)?)|bundles\/[a-f0-9]{64}\.zip)$/.test(
        key,
      )
    ) {
      return new Response("Not found", { status: 404 });
    }
    const immutable = key !== "latest.json";
    // These public, immutable objects don't vary by query string or headers.
    // Keep latest.json out of edge cache so publication is immediately visible.
    const cacheKey = new Request(url.origin + url.pathname);
    const cache = caches.default;
    if (immutable) {
      try {
        const cached = await cache.match(cacheKey);
        if (cached) return clientResponse(request, cached, "HIT", ctx);
      } catch (error) {
        cacheError("edge_cache_read_failed", key, error);
      }
    }
    try {
      const bodyObject =
        request.method === "HEAD"
          ? null
          : await env.STORMSENSE_DATA.get(env.DATA_PREFIX + key);
      const object =
        request.method === "HEAD"
          ? await env.STORMSENSE_DATA.head(env.DATA_PREFIX + key)
          : bodyObject;
      if (!object)
        return Response.json(
          { error: "Data has not been published." },
          { status: 404 },
        );
      const headers = new Headers({
        "Content-Type": key.endsWith(".webp")
          ? "image/webp"
          : key.endsWith(".zip")
            ? "application/zip"
            : key.endsWith(".xml")
              ? "application/xml"
              : "application/json",
        "X-Content-Type-Options": "nosniff",
        "Access-Control-Allow-Origin": "*",
        ETag: object.httpEtag,
        "Content-Length": String(object.size),
        "Cache-Control":
          key === "latest.json"
            ? "no-cache"
            : "public, max-age=31536000, immutable",
      });
      const response = new Response(bodyObject?.body ?? null, {
        headers,
      });
      // Bound the stream tee's possible buffering. Never populate cache from
      // HEAD, missing objects or failures. Cache writes cannot block delivery.
      if (
        immutable &&
        request.method === "GET" &&
        object.size <= 8 * 1024 * 1024
      ) {
        ctx.waitUntil(
          cache
            .put(cacheKey, response.clone())
            .catch((error) =>
              cacheError("edge_cache_write_failed", key, error),
            ),
        );
      }
      return clientResponse(
        request,
        response,
        immutable ? "MISS" : "BYPASS",
        ctx,
      );
    } catch (error) {
      console.error(
        JSON.stringify({
          event: "data_read_failed",
          key,
          error: String(error),
        }),
      );
      return Response.json(
        { error: "Storm data is temporarily unavailable." },
        { status: 503 },
      );
    }
  },
} satisfies ExportedHandler<Env>;

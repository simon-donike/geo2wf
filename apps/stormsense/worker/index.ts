export default {
  async fetch(request, env): Promise<Response> {
    const url = new URL(request.url);
    if (!url.pathname.startsWith("/data/")) return env.ASSETS.fetch(request);
    if (!["GET", "HEAD"].includes(request.method))
      return new Response("Method not allowed", {
        status: 405,
        headers: { Allow: "GET, HEAD" },
      });
    const key = url.pathname.slice("/data/".length);
    if (
      !/^(latest\.json|releases\/[A-Za-z0-9_-]+\/(catalog|coverage|evaluation)\.json|objects\/[a-f0-9]{64}\.json)$/.test(
        key,
      )
    ) {
      return new Response("Not found", { status: 404 });
    }
    try {
      const object = await env.STORMSENSE_DATA.get(env.DATA_PREFIX + key);
      if (!object)
        return Response.json(
          { error: "Data has not been published." },
          { status: 404 },
        );
      const headers = new Headers({
        "Content-Type": "application/json",
        "X-Content-Type-Options": "nosniff",
        ETag: object.httpEtag,
        "Cache-Control":
          key === "latest.json"
            ? "no-cache"
            : "public, max-age=31536000, immutable",
      });
      // GET/HEAD use weak comparison; edge compression can add the W/ prefix.
      const notModified = request.headers
        .get("If-None-Match")
        ?.split(",")
        .some((value) => {
          const tag = value.trim();
          return tag === "*" || tag.replace(/^W\//, "") === object.httpEtag;
        });
      if (notModified)
        return new Response(null, { status: 304, headers });
      return new Response(request.method === "HEAD" ? null : object.body, {
        headers,
      });
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

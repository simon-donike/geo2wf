import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { cpSync, rmSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
export default defineConfig({
  plugins: [
    react(),
    {
      name: "stormsense-data",
      configureServer(server) {
        const dataRoot = resolve(server.config.publicDir, "data");
        // Serve new immutable objects directly: Vite's public-file index does
        // not learn new filenames when the data watcher is intentionally off.
        server.middlewares.use("/data", async (request, response) => {
          response.setHeader("Content-Type", "application/json");
          if (!["GET", "HEAD"].includes(request.method || "")) {
            response.statusCode = 405;
            response.setHeader("Allow", "GET, HEAD");
            response.end(JSON.stringify({ error: "Method not allowed" }));
            return;
          }
          const key = (request.url || "").split("?")[0].replace(/^\//, "");
          if (
            !/^(latest\.json|releases\/[A-Za-z0-9_-]+\/(catalog|coverage|evaluation)\.json|objects\/[a-f0-9]{64}\.json|imagery\/[a-f0-9]{64}\.(?:json|webp(?:\.aux\.xml)?)|bundles\/[a-f0-9]{64}\.zip)$/.test(
              key,
            )
          ) {
            response.statusCode = 404;
            response.end(JSON.stringify({ error: "Not found" }));
            return;
          }
          try {
            const data = await readFile(resolve(dataRoot, key));
            response.setHeader(
              "Content-Type",
              key.endsWith(".webp")
                ? "image/webp"
                : key.endsWith(".zip")
                  ? "application/zip"
                  : key.endsWith(".xml")
                    ? "application/xml"
                    : "application/json",
            );
            response.setHeader(
              "Cache-Control",
              key === "latest.json"
                ? "no-store"
                : "public, max-age=31536000, immutable",
            );
            response.end(request.method === "HEAD" ? undefined : data);
          } catch (error) {
            response.statusCode =
              (error as NodeJS.ErrnoException).code === "ENOENT" ? 404 : 503;
            response.end(
              JSON.stringify({ error: "Storm data is unavailable" }),
            );
          }
        });
      },
      writeBundle(options) {
        const out = options.dir || "dist";
        if (process.env.STORMSENSE_PRODUCTION === "1") {
          // Public data is served from R2. Do not copy the archive just to
          // delete it later, and respect isolated --outDir deployment builds.
          for (const name of ["brand", "land.geojson", "method"]) {
            cpSync(resolve("public", name), resolve(out, name), {
              recursive: true,
            });
          }
        }
        rmSync(resolve(out, "data.publish.lock"), { force: true });
      },
    },
  ],
  server: { port: 5173, watch: { ignored: ["**/public/data/**"] } },
  build: {
    sourcemap: true,
    copyPublicDir: process.env.STORMSENSE_PRODUCTION !== "1",
  },
});

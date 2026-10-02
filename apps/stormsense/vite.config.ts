import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { rmSync } from "node:fs";
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
            !/^(latest\.json|releases\/[A-Za-z0-9_-]+\/(catalog|coverage|evaluation)\.json|objects\/[a-f0-9]{64}\.json)$/.test(
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
      closeBundle() {
        rmSync("dist/data.publish.lock", { force: true });
        if (process.env.STORMSENSE_PRODUCTION === "1")
          rmSync("dist/data", { recursive: true, force: true });
      },
    },
  ],
  server: { port: 5173, watch: { ignored: ["**/public/data/**"] } },
  build: { sourcemap: true },
});

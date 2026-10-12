import manifest from "../public/showcase/v1/manifest.json";
import referenceImages from "../public/reference/sources.json";
import { unzipSync } from "fflate";
import type { DataSource } from "./data";

export { manifest as showcaseManifest };
const assets = new Map<string, Blob>();
const imageUrls = new Map<string, string>();
type Asset = { url: string; bytes: number; sha256: string };

async function verify(blob: Blob, file: Asset) {
  const hash = [
    ...new Uint8Array(
      await crypto.subtle.digest("SHA-256", await blob.arrayBuffer()),
    ),
  ].map((v) => v.toString(16).padStart(2, "0")).join("");
  if (blob.size !== file.bytes || hash !== file.sha256)
    throw Error(`Showcase checksum mismatch: ${file.url}`);
}

async function fetchAsset(file: Asset, signal: AbortSignal) {
  const response = await fetch(file.url, {
    signal: AbortSignal.any([signal, AbortSignal.timeout(60000)]),
    cache: "reload",
  });
  if (!response.ok)
    throw Error(`Could not prepare ${file.url} (${response.status}).`);
  const blob = await response.blob();
  await verify(blob, file);
  return blob;
}

export const SHOWCASE_DATA: DataSource = {
  root: "/showcase/v1/",
  fetch: async (url, init) => {
    init?.signal?.throwIfAborted();
    const blob = assets.get(url);
    if (!blob) throw Error(`Showcase asset is not prepared: ${url}`);
    return new Response(blob);
  },
};

/** Keeps the worked-example images available across route changes, even offline. */
export function showcaseAssetUrl(url: string) {
  const blob = assets.get(url);
  if (!blob) return url;
  if (!imageUrls.has(url)) imageUrls.set(url, URL.createObjectURL(blob));
  return imageUrls.get(url)!;
}

export async function preloadShowcase(
  signal: AbortSignal,
  progress: (done: number, total: number) => void,
) {
  let next = 0;
  let done = 0;
  const files = [...manifest.files, ...referenceImages];
  const packed = new Set<string>();
  const jobs: (() => Promise<void>)[] = [];
  const prepare = async (file: Asset, blob: Blob) => {
    await verify(blob, file);
    signal.throwIfAborted();
    assets.set(file.url, blob);
  };
  for (const pack of manifest.packs) {
    const members = files.slice(pack.start, pack.start + pack.count);
    members.forEach((file) => packed.add(file.url));
    jobs.push(async () => {
      if (members.some((file) => !assets.has(file.url))) {
        const blob = await fetchAsset(pack, signal);
        const entries = unzipSync(new Uint8Array(await blob.arrayBuffer()));
        for (const file of members) {
          signal.throwIfAborted();
          if (assets.has(file.url)) continue;
          const bytes = entries[file.url.slice(SHOWCASE_DATA.root.length)];
          if (!bytes) throw Error(`Could not prepare ${file.url}: missing from pack.`);
          const type = file.url.endsWith(".webp") ? "image/webp"
            : file.url.endsWith(".json") ? "application/json" : "application/xml";
          await prepare(file, new Blob([new Uint8Array(bytes)], { type }));
        }
      }
      progress(done += members.length, files.length);
    });
  }
  for (const file of files.filter((file) => !packed.has(file.url))) {
    jobs.push(async () => {
      if (!assets.has(file.url)) {
        const blob = await fetchAsset(file, signal);
        signal.throwIfAborted();
        assets.set(file.url, blob);
      }
      progress(++done, files.length);
    });
  }
  progress(0, files.length);
  await Promise.all(
    Array.from({ length: 6 }, async () => {
      while (next < jobs.length) {
        signal.throwIfAborted();
        await jobs[next++]();
      }
    }),
  );
}

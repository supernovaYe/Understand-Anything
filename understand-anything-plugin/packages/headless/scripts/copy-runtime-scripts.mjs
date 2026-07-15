import { copyFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const pluginRoot = resolve(packageRoot, "../..");
const sourceRoot = resolve(pluginRoot, "skills/understand");
const outputRoot = resolve(packageRoot, "dist/scripts");
const scripts = [
  "scan-project.mjs",
  "compute-batches.mjs",
  "extract-import-map.mjs",
  "extract-structure.mjs",
  "extract-structure-result.mjs",
];

await mkdir(outputRoot, { recursive: true });
await Promise.all(scripts.map((script) =>
  copyFile(resolve(sourceRoot, script), resolve(outputRoot, script))));

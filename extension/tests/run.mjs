import {build} from "esbuild";
import {mkdtemp, rm} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {pathToFileURL} from "node:url";
const directory = await mkdtemp(join(tmpdir(), "jread-extension-test-"));
try {
  const output = join(directory, "core.test.mjs");
  await build({entryPoints: [new URL("core.test.ts", import.meta.url).pathname], outfile: output, bundle: true, platform: "node", format: "esm"});
  await import(pathToFileURL(output).href);
  await import("./lifecycle.test.mjs");
  await import("./popup.test.mjs");
} finally { await rm(directory, {recursive: true, force: true}); }

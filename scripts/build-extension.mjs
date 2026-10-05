import { build } from 'esbuild';
import { readFile, writeFile } from 'node:fs/promises';
for (const name of ['worker','popup','content','offscreen','worklet']) {
  await build({entryPoints:[`extension/src/${name}.ts`],outfile:`extension/dist/${name}.js`,bundle:true,format:'iife',platform:'browser',target:'chrome116',sourcemap:true});
}
const path = 'extension/manifest.json';
const manifest = JSON.parse(await readFile(path, 'utf8'));
manifest.key = (await readFile('scripts/extension-key.txt','utf8')).trim();
await writeFile(path, JSON.stringify(manifest,null,2)+'\n');

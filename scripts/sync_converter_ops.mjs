import { readdir, readFile, writeFile } from 'node:fs/promises';
import { basename, join } from 'node:path';
import process from 'node:process';

const dir = new URL('../tfjs-converter/src/operations/op_list/', import.meta.url);
const check = process.argv.includes('--check');
const names = (await readdir(dir))
  .filter(name => name.endsWith('.json'))
  .sort();

let stale = 0;
for (const name of names) {
  const jsonUrl = new URL(name, dir);
  const raw = await readFile(jsonUrl, 'utf8');
  JSON.parse(raw);
  const stem = basename(name, '.json');
  const tsUrl = new URL(`${stem}.ts`, dir);
  const next = [
    '/** @license Copyright 2022 Google LLC. Licensed under Apache-2.0. */',
    "import type {OpMapper} from '../types';",
    `const raw = ${JSON.stringify(raw)};`,
    'export const json = JSON.parse(raw) as OpMapper[];',
    '',
  ].join('\n');
  const current = await readFile(tsUrl, 'utf8').catch(() => '');
  if (current === next) continue;
  stale += 1;
  if (!check) await writeFile(tsUrl, next);
}

if (check && stale > 0) {
  console.error(`${stale} converter operation snapshot(s) are stale. Run node scripts/sync_converter_ops.mjs.`);
  process.exit(1);
}

if (!check) console.log(`Synced ${names.length} converter operation snapshot(s).`);

#!/usr/bin/env node

import {execFileSync} from 'node:child_process';
import {readFileSync} from 'node:fs';
import {basename} from 'node:path';

const tracked = execFileSync('git', ['ls-files', '-z'], {
  encoding: 'utf8'
}).split('\0').filter(Boolean);

const forbiddenExtensions = /\.(?:py|pyi|pyx|pxd|pxi)$/i;
const forbiddenPathComponent = /(?:^|\/)(?:python|pypi)(?:\/|$)/i;
const forbiddenBasenames = new Set([
  'setup.py',
  'pyproject.toml',
  'pipfile',
  'pipfile.lock',
  'tox.ini'
]);
const forbiddenRequirement = /^requirements(?:-[^/]+)?\.txt$/i;
const forbiddenCommand = /(?:^|[\s/_.-])(?:python\d*|pypi|twine|pip\d*|build-pip-package)(?:$|[\s/_.-])/i;
const forbiddenDependency = /(?:^|[-_/])(?:python|pypi|twine)(?:$|[-_/])/i;

const violations = [];

for (const path of tracked) {
  const file = basename(path);
  const lowerFile = file.toLowerCase();

  if (forbiddenExtensions.test(path) ||
      forbiddenPathComponent.test(path) ||
      forbiddenBasenames.has(lowerFile) ||
      forbiddenRequirement.test(file)) {
    violations.push(`${path}: forbidden Python source or project path`);
  }

  if (!path.endsWith('package.json')) {
    continue;
  }

  const manifest = JSON.parse(readFileSync(path, 'utf8'));
  for (const [name, command] of Object.entries(manifest.scripts ?? {})) {
    if (forbiddenCommand.test(String(command))) {
      violations.push(`${path}: script ${name} invokes a Python or PyPI workflow`);
    }
  }

  for (const section of [
    'dependencies',
    'devDependencies',
    'optionalDependencies',
    'peerDependencies'
  ]) {
    for (const name of Object.keys(manifest[section] ?? {})) {
      if (forbiddenDependency.test(name)) {
        violations.push(`${path}: ${section} contains ${name}`);
      }
    }
  }
}

if (violations.length > 0) {
  console.error('Source policy violations:');
  for (const violation of violations) {
    console.error(`- ${violation}`);
  }
  process.exit(1);
}

console.log('Source policy passed: no tracked Python source or Python/PyPI workflow.');

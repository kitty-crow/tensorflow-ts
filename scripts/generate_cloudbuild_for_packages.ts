#!/usr/bin/env node
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {findPackagesWithDiff, allPackages} from './find_packages_with_diff.js';
import {generateCloudbuild} from './generate_cloudbuild';
import {ArgumentParser} from 'argparse';

import * as fs from 'fs';
import {requireRecord, requireString, requireStringArray} from './release-inputs';

const parser = new ArgumentParser({
  description: 'Generate a cloudbuild file to test packages. When run' +
      ' with no arguments, tests packages affected by the current' +
      ' changes.'
});

parser.addArgument('packages', {
  type: String,
  nargs: '*',
  help: 'packages to consider as having changed',
});

parser.addArgument(['-o', '--output'], {
  type: String,
  nargs: '?',
  defaultValue: 'cloudbuild_generated.yml',
});

const raw = requireRecord(parser.parseArgs(process.argv.slice(2)), 'cloudbuild arguments');
const args = {packages: requireStringArray(raw.packages, 'packages'), output: requireString(raw.output, 'output')};

let packages: string[];
const nightly = Boolean(process.env['NIGHTLY']);
if (args.packages.length > 0) {
  // Test packages specified in command line args.
  packages = args.packages;
} else if (nightly) {
  // Test all packages during the nightly build.
  packages = allPackages;
} else {
  // Test packages that have changed.
  packages = findPackagesWithDiff();
}

const cloudbuild = generateCloudbuild(packages, nightly);
// The repository uses js-yaml 3.x, whereas its declaration package describes
// 4.x. Preserve the original safe serializer rather than allowing JS tags.
const yamlRuntime = requireRecord(require('js-yaml'), 'js-yaml 3.x');
const safeDump = yamlRuntime.safeDump;
if (typeof safeDump !== 'function') {
  throw new Error('js-yaml 3.x safeDump is required');
}
const output = requireString(safeDump(cloudbuild), 'Cloud Build YAML');
fs.writeFileSync(args.output, output);

// Copyright 2020 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

import { generateCloudbuild, CloudbuildYaml} from './generate_cloudbuild';
//const generateCloudbuild = require('./generate_cloudbuild').generateCloudbuild;
import * as yaml from 'js-yaml';
import * as fs from 'fs';
import * as path from 'path';
import {findSubgraph, parseDependencyGraph, topologicalSort, transposeGraph} from './graph_utils';

// These tests only detect changes in the generated file, and may need to be
// updated if the project structure changes. To update them, run
// 'yarn update-cloudbuild-tests`.
// TODO(mattsoulanille): When Jasmine is updated to >=3.3.0, Use
// jasmine.withContext to show the above message if the tests fail.
describe('generateCloudbuild', () => {
  it('generates the correct cloudbuild file for e2e', () => {
    const expectedCloudbuild = yaml.load(fs.readFileSync(
      path.join(__dirname, 'cloudbuild_e2e_expected.yml'), 'utf8'));
    const cloudbuild = generateCloudbuild(['e2e'], /* nightly */ false,
                                          /* print */ false);
    expect(cloudbuild).toEqual(expectedCloudbuild as CloudbuildYaml);
  });

  it('generates the correct cloudbuild file for tfjs-node', () => {
    const expectedCloudbuild = yaml.load(fs.readFileSync(
      path.join(__dirname, 'cloudbuild_tfjs_node_expected.yml'), 'utf8'));
    const cloudbuild = generateCloudbuild(['tfjs-node'], /* nightly */ false,
                                          /* print */ false);
    expect(cloudbuild).toEqual(expectedCloudbuild as CloudbuildYaml);
  });
});


describe('dependency graph contracts', () => {
  it('preserves traversal and reverse dependency order', () => {
    const graph = parseDependencyGraph('{"a":["b","c"],"b":["c"],"c":[]}');
    expect([...findSubgraph('a', graph)]).toEqual(['b', 'c']);
    expect(topologicalSort(graph)).toEqual(['c', 'b', 'a']);
    expect([...findSubgraph('c', transposeGraph(graph))]).toEqual(['a', 'b']);
    expect([...findSubgraph('missing', graph)]).toEqual([]);
  });

  it('rejects malformed JSON graph shapes and missing edges', () => {
    expect(() => parseDependencyGraph('[]')).toThrowError('Dependency graph must be an object');
    expect(() => parseDependencyGraph('{"a":[1]}')).toThrowError('Dependency graph edges for a must be strings');
    expect(() => parseDependencyGraph('{"a":["b"]}')).toThrowError('Graph edge b of node a not found in the graph');
  });
});

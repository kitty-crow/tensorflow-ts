/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {ClipVec4Program} from './clip_vec4_webgpu';
import {FillProgram} from './fill_webgpu';
import {OneHotProgram} from './onehot_webgpu';
import {ReduceProgram} from './reduce_webgpu';
import {ResizeBilinearProgram} from './resize_bilinear_webgpu';
import {MergeProgram, SwapProgram} from './top_k_webgpu';

describe('flat WebGPU program dispatch', () => {
  it('retains shape identity and scalar and empty dispatches', () => {
    const shape: number[] = [];
    const scalar = new FillProgram(shape);
    expect(scalar.outputShape).toBe(shape);
    expect(scalar.dispatchLayout).toEqual({x: []});
    expect(scalar.dispatch).toEqual([1, 1, 1]);
    expect(new FillProgram([0, 3]).dispatch).toEqual([0, 1, 1]);
    expect(scalar.variableNames).toEqual([]);
    expect(scalar.size).toBe(true);
  });

  it('uses the supplied workgroup and vector width for dispatch', () => {
    const program = new ClipVec4Program([257]);
    expect(program.dispatch).toEqual([2, 1, 1]);
    expect(program.workgroupSize).toEqual([64, 1, 1]);
    expect(program.workPerThread).toBe(4);
    expect(program.outputComponent).toBe(4);
    expect(program.variableNames).toEqual(['A']);
    expect(program.shaderKey).toBe('clipVec4');
  });

  it('keeps mutable metadata separate per instance', () => {
    const first = new FillProgram([65]);
    const second = new FillProgram([65]);
    first.variableNames.push('test');
    first.workgroupSize[0] = 1;
    first.dispatchLayout.x.push(1);
    first.dispatch[0] = 7;
    expect(second.variableNames).toEqual([]);
    expect(second.workgroupSize).toEqual([64, 1, 1]);
    expect(second.dispatchLayout).toEqual({x: [0]});
    expect(second.dispatch).toEqual([2, 1, 1]);
  });

  it('preserves operation-specific metadata and shader text', () => {
    const oneHot = new OneHotProgram(5, 13);
    expect(oneHot.outputShape).toEqual([5, 13]);
    expect(oneHot.dispatch).toEqual([2, 1, 1]);
    expect(oneHot.shaderKey).toBe('onehot');
    expect(oneHot.getUserCode()).toContain('uniforms.offValue');
    const resize = new ResizeBilinearProgram([1, 2, 3, 4], 5, 6);
    expect(resize.outputShape).toEqual([1, 5, 6, 4]);
    expect(resize.dispatch).toEqual([2, 1, 1]);
    expect(resize.shaderKey).toBe('resizeBilinear');
  });

  it('preserves distinct top-k passes and their workgroup geometry', () => {
    const shape = [1, 257];
    const swap = new SwapProgram(shape);
    const merge = new MergeProgram(shape);
    expect(swap.outputShape).toBe(shape);
    expect(merge.outputShape).toBe(shape);
    expect(swap.dispatch).toEqual([2, 1, 1]);
    expect(merge.dispatch).toEqual([2, 1, 1]);
    expect(swap.shaderKey).toBe('swap');
    expect(merge.shaderKey).toBe('merge');
    expect(swap.getUserCode()).not.toBe(merge.getUserCode());
  });
});

describe('reduction dispatch selection', () => {
  it('preserves workgroup boundaries and the device limit', () => {
    expect(new ReduceProgram({batchSize: 1, inSize: 4095, outSize: 1, windowSize: 4095}, 'sum', 512)
        .workgroupSize).toEqual([64, 1, 1]);
    expect(new ReduceProgram({batchSize: 1, inSize: 4096, outSize: 1, windowSize: 4096}, 'sum', 512)
        .workgroupSize).toEqual([256, 1, 1]);
    expect(new ReduceProgram({batchSize: 1, inSize: 32768, outSize: 1, windowSize: 32768}, 'sum', 256)
        .workgroupSize).toEqual([256, 1, 1]);
    expect(new ReduceProgram({batchSize: 1, inSize: 32768, outSize: 1, windowSize: 32768}, 'sum', 512)
        .workgroupSize).toEqual([512, 1, 1]);
  });

  it('preserves operation-specific reduction code and initial values', () => {
    const info = {batchSize: 2, inSize: 8, outSize: 1, windowSize: 8};
    expect(new ReduceProgram(info, 'min', 256).getUserCode())
        .toContain('candidate < bestValue');
    expect(new ReduceProgram(info, 'max', 256).getUserCode())
        .toContain('candidate > bestValue');
    expect(new ReduceProgram(info, 'sum', 256).getUserCode())
        .toContain('bestValue = bestValue + candidate');
    expect(new ReduceProgram(info, 'mean', 256).getUserCode())
        .toContain('bestValue / f32(uniforms.reduceSize)');
    expect(new ReduceProgram(info, 'prod', 256).getUserCode())
        .toContain('bestValue = bestValue * candidate');
    expect(new ReduceProgram(info, 'all', 256).getUserCode())
        .toContain('bestValue >= 1.0 && candidate >= 1.0');
    expect(new ReduceProgram(info, 'any', 256).getUserCode())
        .toContain('bestValue >= 1.0 || candidate >= 1.0');
  });
});

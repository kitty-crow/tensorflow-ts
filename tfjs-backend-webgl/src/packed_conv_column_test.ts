/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {selectActivationSource} from './activation_source';
import {packedConvColumnSource} from './packed_conv_column';

describe('packed convolution sampling source', () => {
  it('does not emit a sample beyond the filter boundary', () => {
    expect(packedConvColumnSource(4, 4, 1, 0, 1)).toBe('');
    expect(packedConvColumnSource(4, 4, 2, 1, 2)).toBe('');
  });

  it('reuses the adjacent texel for odd padding and unit dilation', () => {
    const code = packedConvColumnSource(2, 4, 1, 1, 1);
    expect(code).toContain('xC2 = vec4(xTexelC0.zw, xTexelC2.xy)');
    expect(code).toContain('xC3 = xTexelC2;');
    expect(code).not.toContain('previous = getX');
  });

  it('samples the previous texel for the first odd-padded column', () => {
    const code = packedConvColumnSource(0, 1, 1, 1, 1);
    expect(code).toContain('xCOffset = xC + 1 - 2;');
    expect(code).toContain('previous = getX');
    expect(code).not.toContain('xC1 =');
  });

  it('uses separate texel reads when dilation prevents sharing', () => {
    const code = packedConvColumnSource(0, 2, 1, 1, 2);
    expect(code).toContain('xCOffset = xC + imod(pads[1], 2) + 2;');
    expect(code).toContain('xCOffset -= 2;');
    expect(code).toContain('xC1 = vec4(previous.zw, xTexelC1.xy)');
  });

  it('retains stride-two odd and even padding paths', () => {
    const odd = packedConvColumnSource(0, 2, 2, 1, 1);
    expect(odd).toContain('xCOffset = xC + 1 - strides[1];');
    expect(odd).toContain('final = getX');
    expect(odd).toContain('xC1 = vec4(xTexelC1.xy, final.xy);');
    const even = packedConvColumnSource(0, 2, 2, 0, 1);
    expect(even).toContain('xCOffset = xC + strides[1];');
    expect(even).toContain('xC1 = vec4(xTexelC0.zw, xTexelC1.zw);');
    expect(even).not.toContain('final = getX');
  });

  it('does not emit an unused second output column for odd filter width', () => {
    expect(packedConvColumnSource(2, 3, 1, 0, 1)).not.toContain('xC3 =');
    expect(packedConvColumnSource(2, 3, 2, 1, 1)).not.toContain('xC3 =');
    expect(packedConvColumnSource(2, 3, 2, 0, 1)).not.toContain('xC3 =');
  });
});

describe('ordered fused activation source', () => {
  it('only evaluates the selected fragment, retaining PReLU precedence', () => {
    const prelu = jasmine.createSpy('prelu').and.returnValue('prelu');
    const leaky = jasmine.createSpy('leaky').and.returnValue('leaky');
    const plain = jasmine.createSpy('plain').and.returnValue('plain');
    expect(selectActivationSource('', true, true, prelu, leaky, plain)).toBe('');
    expect(prelu).not.toHaveBeenCalled();
    expect(leaky).not.toHaveBeenCalled();
    expect(plain).not.toHaveBeenCalled();
    expect(selectActivationSource('a', true, true, prelu, leaky, plain))
        .toBe('prelu');
    expect(prelu).toHaveBeenCalledTimes(1);
    expect(leaky).not.toHaveBeenCalled();
    expect(plain).not.toHaveBeenCalled();
    expect(selectActivationSource('a', false, true, prelu, leaky, plain))
        .toBe('leaky');
    expect(selectActivationSource('a', false, false, prelu, leaky, plain))
        .toBe('plain');
    expect(leaky).toHaveBeenCalledTimes(1);
    expect(plain).toHaveBeenCalledTimes(1);
  });
});

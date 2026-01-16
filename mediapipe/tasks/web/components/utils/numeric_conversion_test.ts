/**
 * Copyright 2025 The MediaPipe Authors.
 *
 * <p>Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * <p>http://www.apache.org/licenses/LICENSE-2.0
 *
 * <p>Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

import 'jasmine';

import {asLegacyNumberOrString} from '../../../../tasks/web/components/utils/numeric_conversion';

describe('NumericConversion', () => {
  it('Converts safe integer values to numbers', () => {
    expect(asLegacyNumberOrString(1)).toBe(1);
    expect(asLegacyNumberOrString(0)).toBe(0);
    expect(asLegacyNumberOrString(Number.MAX_SAFE_INTEGER)).toBe(
      Number.MAX_SAFE_INTEGER,
    );
    expect(asLegacyNumberOrString(Number.MIN_SAFE_INTEGER)).toBe(
      Number.MIN_SAFE_INTEGER,
    );

    expect(asLegacyNumberOrString('1')).toBe(1);
    expect(asLegacyNumberOrString('0')).toBe(0);
    expect(asLegacyNumberOrString(Number.MAX_SAFE_INTEGER.toString())).toBe(
      Number.MAX_SAFE_INTEGER,
    );
    expect(asLegacyNumberOrString(Number.MIN_SAFE_INTEGER.toString())).toBe(
      Number.MIN_SAFE_INTEGER,
    );
  });

  it('Converts unsafe integer values to string', () => {
    const int64Max = '9223372036854775807';
    const int64Min = '-9223372036854775808';
    expect(asLegacyNumberOrString(int64Max)).toBe(
      int64Max as unknown as number,
    );
    expect(asLegacyNumberOrString(int64Min)).toBe(
      int64Min as unknown as number,
    );
    expect(asLegacyNumberOrString(Number.MAX_VALUE)).toBe(
      Number.MAX_VALUE.toString() as unknown as number,
    );
  });
});

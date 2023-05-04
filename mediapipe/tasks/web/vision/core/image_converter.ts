/**
 * Copyright 2023 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {MPImageChannelConverter, RGBAColor} from '../../../../tasks/web/vision/core/image';

/**
 * Color converter that falls back to a default implementation if the
 * user-provided converter does not specify a conversion.
 */
export class DefaultColorConverter implements
    Required<MPImageChannelConverter> {
  private static readonly WARNINGS_LOGGED = new Set<string>();

  constructor(private readonly customConverter: MPImageChannelConverter) {}

  floatToRGBAConverter(v: number): RGBAColor {
    if (this.customConverter.floatToRGBAConverter) {
      return this.customConverter.floatToRGBAConverter(v);
    }
    this.logWarningOnce('floatToRGBAConverter');
    return [v * 255, v * 255, v * 255, 255];
  }

  uint8ToRGBAConverter(v: number): RGBAColor {
    if (this.customConverter.uint8ToRGBAConverter) {
      return this.customConverter.uint8ToRGBAConverter(v);
    }
    this.logWarningOnce('uint8ToRGBAConverter');
    return [v, v, v, 255];
  }

  rgbaToFloatConverter(r: number, g: number, b: number, a: number): number {
    if (this.customConverter.rgbaToFloatConverter) {
      return this.customConverter.rgbaToFloatConverter(r, g, b, a);
    }
    this.logWarningOnce('rgbaToFloatConverter');
    return (r / 3 + g / 3 + b / 3) / 255;
  }

  rgbaToUint8Converter(r: number, g: number, b: number, a: number): number {
    if (this.customConverter.rgbaToUint8Converter) {
      return this.customConverter.rgbaToUint8Converter(r, g, b, a);
    }
    this.logWarningOnce('rgbaToUint8Converter');
    return r / 3 + g / 3 + b / 3;
  }

  floatToUint8Converter(v: number): number {
    if (this.customConverter.floatToUint8Converter) {
      return this.customConverter.floatToUint8Converter(v);
    }
    this.logWarningOnce('floatToUint8Converter');
    return v * 255;
  }

  uint8ToFloatConverter(v: number): number {
    if (this.customConverter.uint8ToFloatConverter) {
      return this.customConverter.uint8ToFloatConverter(v);
    }
    this.logWarningOnce('uint8ToFloatConverter');
    return v / 255;
  }

  private logWarningOnce(methodName: string): void {
    if (!DefaultColorConverter.WARNINGS_LOGGED.has(methodName)) {
      console.log(`Using default ${methodName}`);
      DefaultColorConverter.WARNINGS_LOGGED.add(methodName);
    }
  }
}

/**
 * Copyright 2022 The MediaPipe Authors.
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

import 'jasmine';

import {EmbedderOptions as EmbedderOptionsProto} from '../../../../tasks/cc/components/processors/proto/embedder_options_pb';
import {EmbedderOptions} from '../../../../tasks/web/core/embedder_options';

import {convertEmbedderOptionsToProto} from './embedder_options';

interface TestCase {
  optionName: keyof EmbedderOptions;
  protoName: string;
  customValue: unknown;
  defaultValue: unknown;
}

describe('convertEmbedderOptionsToProto()', () => {
  function verifyOption(
      actualEmbedderOptions: EmbedderOptionsProto,
      expectedEmbedderOptions: Record<string, unknown> = {}): void {
    expect(actualEmbedderOptions.toObject())
        .toEqual(jasmine.objectContaining(expectedEmbedderOptions));
  }

  const testCases: TestCase[] = [
    {
      optionName: 'l2Normalize',
      protoName: 'l2Normalize',
      customValue: true,
      defaultValue: undefined
    },
    {
      optionName: 'quantize',
      protoName: 'quantize',
      customValue: true,
      defaultValue: undefined
    },
  ];

  for (const testCase of testCases) {
    it(`can set ${testCase.optionName}`, () => {
      const embedderOptionsProto = convertEmbedderOptionsToProto(
          {[testCase.optionName]: testCase.customValue});
      verifyOption(
          embedderOptionsProto, {[testCase.protoName]: testCase.customValue});
    });

    it(`can clear ${testCase.optionName}`, () => {
      let embedderOptionsProto = convertEmbedderOptionsToProto(
          {[testCase.optionName]: testCase.customValue});
      verifyOption(
          embedderOptionsProto, {[testCase.protoName]: testCase.customValue});

      embedderOptionsProto =
          convertEmbedderOptionsToProto({[testCase.optionName]: undefined});
      verifyOption(
          embedderOptionsProto, {[testCase.protoName]: testCase.defaultValue});
    });
  }

  it('overwrites options', () => {
    let embedderOptionsProto =
        convertEmbedderOptionsToProto({l2Normalize: true});
    verifyOption(embedderOptionsProto, {'l2Normalize': true});

    embedderOptionsProto = convertEmbedderOptionsToProto(
        {l2Normalize: false}, embedderOptionsProto);
    verifyOption(embedderOptionsProto, {'l2Normalize': false});
  });

  it('replaces options', () => {
    let embedderOptionsProto = convertEmbedderOptionsToProto({quantize: true});
    verifyOption(embedderOptionsProto, {'quantize': true});

    embedderOptionsProto = convertEmbedderOptionsToProto(
        {l2Normalize: true}, embedderOptionsProto);
    verifyOption(embedderOptionsProto, {'l2Normalize': true, 'quantize': true});
  });
});

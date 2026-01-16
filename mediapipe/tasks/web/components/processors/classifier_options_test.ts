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

import {ClassifierOptions as ClassifierOptionsProto} from '../../../../tasks/cc/components/processors/proto/classifier_options_pb';
import {ClassifierOptions} from '../../../../tasks/web/core/classifier_options';

import {convertClassifierOptionsToProto} from './classifier_options';

interface TestCase {
  optionName: keyof ClassifierOptions;
  protoName: string;
  customValue: unknown;
  defaultValue: unknown;
}

describe('convertClassifierOptionsToProto()', () => {
  function verifyOption(
      actualClassifierOptions: ClassifierOptionsProto,
      expectedClassifierOptions: Record<string, unknown> = {}): void {
    expect(actualClassifierOptions.toObject())
        .toEqual(jasmine.objectContaining(expectedClassifierOptions));
  }

  const testCases: TestCase[] = [
    {
      optionName: 'maxResults',
      protoName: 'maxResults',
      customValue: 5,
      defaultValue: -1
    },
    {
      optionName: 'displayNamesLocale',
      protoName: 'displayNamesLocale',
      customValue: 'en',
      defaultValue: 'en'
    },
    {
      optionName: 'scoreThreshold',
      protoName: 'scoreThreshold',
      customValue: 0.1,
      defaultValue: undefined
    },
    {
      optionName: 'categoryAllowlist',
      protoName: 'categoryAllowlistList',
      customValue: ['foo'],
      defaultValue: []
    },
    {
      optionName: 'categoryDenylist',
      protoName: 'categoryDenylistList',
      customValue: ['bar'],
      defaultValue: []
    },
  ];

  for (const testCase of testCases) {
    it(`can set ${testCase.optionName}`, () => {
      const classifierOptionsProto = convertClassifierOptionsToProto(
          {[testCase.optionName]: testCase.customValue});
      verifyOption(
          classifierOptionsProto, {[testCase.protoName]: testCase.customValue});
    });

    it(`can clear ${testCase.optionName}`, () => {
      let classifierOptionsProto = convertClassifierOptionsToProto(
          {[testCase.optionName]: testCase.customValue});
      verifyOption(
          classifierOptionsProto, {[testCase.protoName]: testCase.customValue});

      classifierOptionsProto =
          convertClassifierOptionsToProto({[testCase.optionName]: undefined});
      verifyOption(
          classifierOptionsProto,
          {[testCase.protoName]: testCase.defaultValue});
    });
  }

  it('overwrites options', () => {
    let classifierOptionsProto =
        convertClassifierOptionsToProto({maxResults: 1});
    verifyOption(classifierOptionsProto, {'maxResults': 1});

    classifierOptionsProto = convertClassifierOptionsToProto(
        {maxResults: 2}, classifierOptionsProto);
    verifyOption(classifierOptionsProto, {'maxResults': 2});
  });

  it('merges options', () => {
    let classifierOptionsProto =
        convertClassifierOptionsToProto({maxResults: 1});
    verifyOption(classifierOptionsProto, {'maxResults': 1});

    classifierOptionsProto = convertClassifierOptionsToProto(
        {displayNamesLocale: 'en'}, classifierOptionsProto);
    verifyOption(
        classifierOptionsProto, {'maxResults': 1, 'displayNamesLocale': 'en'});
  });
});

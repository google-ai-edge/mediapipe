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

import {ClassifierOptions as ClassifierOptionsProto} from '../../../../tasks/cc/components/processors/proto/classifier_options_pb';
import {ClassifierOptions} from '../../../../tasks/web/core/classifier_options';

/**
 * Converts a ClassifierOptions object to its Proto representation, optionally
 * based on existing definition.
 * @param options The options object to convert to a Proto. Only options that
 *     are expliclty provided are set.
 * @param baseOptions A base object that options can be merged into.
 */
export function convertClassifierOptionsToProto(
    options: ClassifierOptions,
    baseOptions?: ClassifierOptionsProto): ClassifierOptionsProto {
  const classifierOptions =
      baseOptions ? baseOptions.clone() : new ClassifierOptionsProto();
  if (options.displayNamesLocale !== undefined) {
    classifierOptions.setDisplayNamesLocale(options.displayNamesLocale);
  } else if (options.displayNamesLocale === undefined) {
    classifierOptions.clearDisplayNamesLocale();
  }

  if (options.maxResults !== undefined) {
    classifierOptions.setMaxResults(options.maxResults);
  } else if ('maxResults' in options) {  // Check for undefined
    classifierOptions.clearMaxResults();
  }

  if (options.scoreThreshold !== undefined) {
    classifierOptions.setScoreThreshold(options.scoreThreshold);
  } else if ('scoreThreshold' in options) {  // Check for undefined
    classifierOptions.clearScoreThreshold();
  }

  if (options.categoryAllowlist !== undefined) {
    classifierOptions.setCategoryAllowlistList(options.categoryAllowlist);
  } else if ('categoryAllowlist' in options) {  // Check for undefined
    classifierOptions.clearCategoryAllowlistList();
  }

  if (options.categoryDenylist !== undefined) {
    classifierOptions.setCategoryDenylistList(options.categoryDenylist);
  } else if ('categoryDenylist' in options) {  // Check for undefined
    classifierOptions.clearCategoryDenylistList();
  }
  return classifierOptions;
}

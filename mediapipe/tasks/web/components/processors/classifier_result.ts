/**
 * Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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

import {ClassificationEntry as ClassificationEntryProto, ClassificationResult} from '../../../../tasks/cc/components/containers/proto/classifications_pb';
import {ClassificationEntry, Classifications} from '../../../../tasks/web/components/containers/classifications';

const DEFAULT_INDEX = -1;
const DEFAULT_SCORE = 0.0;

/**
 * Converts a ClassificationEntry proto to the ClassificationEntry result
 * type.
 */
function convertFromClassificationEntryProto(source: ClassificationEntryProto):
    ClassificationEntry {
  const categories = source.getCategoriesList().map(category => {
    return {
      index: category.getIndex() ?? DEFAULT_INDEX,
      score: category.getScore() ?? DEFAULT_SCORE,
      displayName: category.getDisplayName() ?? '',
      categoryName: category.getCategoryName() ?? '',
    };
  });

  return {
    categories,
    timestampMs: source.getTimestampMs(),
  };
}

/**
 * Converts a ClassificationResult proto to a list of classifications.
 */
export function convertFromClassificationResultProto(
    classificationResult: ClassificationResult) : Classifications[] {
  const result: Classifications[] = [];
  for (const classificationsProto of
           classificationResult.getClassificationsList()) {
    const classifications: Classifications = {
      entries: classificationsProto.getEntriesList().map(
          entry => convertFromClassificationEntryProto(entry)),
      headIndex: classificationsProto.getHeadIndex() ?? DEFAULT_INDEX,
      headName: classificationsProto.getHeadName() ?? '',
    };
    result.push(classifications);
  }
  return result;
}

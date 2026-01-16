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

import {Classification as ClassificationProto} from '../../../../framework/formats/classification_pb';
import {
  ClassificationResult as ClassificationResultProto,
  Classifications as ClassificationsProto,
} from '../../../../tasks/cc/components/containers/proto/classifications_pb';
import {
  ClassificationResult,
  Classifications,
} from '../../../../tasks/web/components/containers/classification_result';
import {asLegacyNumberOrString} from '../../../../tasks/web/components/utils/numeric_conversion';

const DEFAULT_INDEX = -1;
const DEFAULT_SCORE = 0.0;

/**
 * Converts a list of Classification protos to a Classifications object.
 */
export function convertFromClassifications(
  classifications: readonly ClassificationProto[],
  headIndex = DEFAULT_INDEX,
  headName = '',
): Classifications {
  const categories = classifications.map((classification) => {
    return {
      index: classification.getIndex() ?? DEFAULT_INDEX,
      score: classification.getScore() ?? DEFAULT_SCORE,
      categoryName: classification.getLabel() ?? '',
      displayName: classification.getDisplayName() ?? '',
    };
  });
  return {
    categories,
    headIndex,
    headName,
  };
}

/**
 * Converts a Classifications proto to a Classifications object.
 */
function convertFromClassificationsProto(
  source: ClassificationsProto,
): Classifications {
  return convertFromClassifications(
    source.getClassificationList()?.getClassificationList() ?? [],
    source.getHeadIndex(),
    source.getHeadName(),
  );
}

/**
 * Converts a ClassificationResult proto to a ClassificationResult object.
 */
export function convertFromClassificationResultProto(
  source: ClassificationResultProto,
): ClassificationResult {
  const result: ClassificationResult = {
    classifications: source
      .getClassificationsList()
      .map((classififications) =>
        convertFromClassificationsProto(classififications),
      ),
  };
  if (source.hasTimestampMs()) {
    result.timestampMs = asLegacyNumberOrString(source.getTimestampMs());
  }
  return result;
}

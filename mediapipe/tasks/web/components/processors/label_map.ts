/**
 * Copyright 2026 The MediaPipe Authors.
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

import {CalculatorGraphConfig} from '../../../../framework/calculator_pb';
import {LabelMapItem} from '../../../../util/label_map_pb';

/** Category names and locale display names, aligned by class index. */
export interface CategoryLabelMap {
  labels: string[];
  displayNames: string[];
}

/**
 * Converts a TFLite / MediaPipe label_items map into parallel arrays.
 *
 * Index `i` is `categoryName` / `displayName` for class id `i`. Missing
 * display names become an empty string so callers can zip the arrays.
 */
export function categoryLabelMapFromItems(
  labelItems: {
    forEach: (
      callback: (value: LabelMapItem, key: string | number) => void,
    ) => void;
  },
): CategoryLabelMap {
  const labels: string[] = [];
  const displayNames: string[] = [];
  labelItems.forEach((value, index) => {
    const i = Number(index);
    labels[i] = value.getName() ?? '';
    displayNames[i] = value.getDisplayName() ?? '';
  });
  return {labels, displayNames};
}

/**
 * Finds expanded-graph nodes whose name or calculator type contains
 * `calculatorName`. Throws if more than one node matches.
 */
export function findUniqueCalculatorNode(
  graphConfig: CalculatorGraphConfig,
  calculatorName: string,
): CalculatorGraphConfig.Node | undefined {
  const nodes = graphConfig.getNodeList().filter(
    (n: CalculatorGraphConfig.Node) =>
      n.getName().includes(calculatorName) ||
      n.getCalculator().includes(calculatorName),
  );
  if (nodes.length > 1) {
    throw new Error(`The graph has more than one ${calculatorName}.`);
  }
  return nodes[0];
}

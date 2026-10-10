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

import 'jasmine';

import {CalculatorGraphConfig} from '../../../../framework/calculator_pb';
import {LabelMapItem} from '../../../../util/label_map_pb';

import {
  categoryLabelMapFromItems,
  findUniqueCalculatorNode,
} from './label_map';

describe('categoryLabelMapFromItems', () => {
  it('aligns names and display names by class index', () => {
    const items = new Map<number, LabelMapItem>();
    const apple = new LabelMapItem();
    apple.setName('apple');
    apple.setDisplayName('manzana');
    items.set(0, apple);
    const banana = new LabelMapItem();
    banana.setName('banana');
    banana.setDisplayName('plátano');
    items.set(1, banana);

    expect(categoryLabelMapFromItems(items)).toEqual({
      labels: ['apple', 'banana'],
      displayNames: ['manzana', 'plátano'],
    });
  });

  it('returns no display names when the labelmap has none', () => {
    const items = new Map<number, LabelMapItem>();
    const cat = new LabelMapItem();
    cat.setName('cat');
    items.set(0, cat);
    const dog = new LabelMapItem();
    dog.setName('dog');
    items.set(1, dog);

    expect(categoryLabelMapFromItems(items)).toEqual({
      labels: ['cat', 'dog'],
      displayNames: [],
    });
  });

  it('returns empty arrays for an empty map', () => {
    expect(categoryLabelMapFromItems(new Map())).toEqual({
      labels: [],
      displayNames: [],
    });
  });
});

describe('findUniqueCalculatorNode', () => {
  it('returns undefined when the graph has no matching node', () => {
    expect(
      findUniqueCalculatorNode(
        new CalculatorGraphConfig(),
        'TensorsToClassificationCalculator',
      ),
    ).toBeUndefined();
  });

  it('throws when more than one node matches', () => {
    const graph = new CalculatorGraphConfig();
    const a = new CalculatorGraphConfig.Node();
    a.setCalculator('TensorsToClassificationCalculator');
    const b = new CalculatorGraphConfig.Node();
    b.setName('TensorsToClassificationCalculator');
    graph.addNode(a);
    graph.addNode(b);

    expect(() =>
      findUniqueCalculatorNode(graph, 'TensorsToClassificationCalculator'),
    ).toThrowError(/more than one/);
  });
});

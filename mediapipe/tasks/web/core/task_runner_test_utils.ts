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

import {CalculatorGraphConfig} from '../../../framework/calculator_pb';
import {CALCULATOR_GRAPH_CONFIG_LISTENER_NAME, SimpleListener, WasmModule} from '../../../web/graph_runner/graph_runner';
import {WasmModuleRegisterModelResources} from '../../../web/graph_runner/register_model_resources_graph_service';

type SpyWasmModuleInternal = WasmModule&WasmModuleRegisterModelResources;

/**
 * Convenience type for our fake WasmModule for Jasmine testing.
 */
export declare type SpyWasmModule = jasmine.SpyObj<SpyWasmModuleInternal>;

/**
 * Factory function for creating a fake WasmModule for our Jasmine tests,
 * allowing our APIs to no longer rely on the Wasm layer so they can run tests
 * in pure JS/TS (and optionally spy on the calls).
 */
export function createSpyWasmModule(): SpyWasmModule {
  const spyWasmModule = jasmine.createSpyObj<SpyWasmModuleInternal>([
    '_setAutoRenderToScreen', 'stringToNewUTF8', '_attachProtoListener',
    '_attachProtoVectorListener', '_free', '_waitUntilIdle',
    '_addStringToInputStream', '_registerModelResourcesGraphService',
    '_configureAudio', '_malloc', '_addProtoToInputStream', '_getGraphConfig',
    '_closeGraph', '_addBoolToInputStream'
  ]);
  spyWasmModule._getGraphConfig.and.callFake(() => {
    (spyWasmModule.simpleListeners![CALCULATOR_GRAPH_CONFIG_LISTENER_NAME] as
     SimpleListener<Uint8Array>)(
        new CalculatorGraphConfig().serializeBinary(), 0);
  });
  spyWasmModule.HEAPU8 = jasmine.createSpyObj<Uint8Array>(['set']);
  return spyWasmModule;
}

/**
 * Sets up our equality testing to use a custom float equality checking function
 * to avoid incorrect test results due to minor floating point inaccuracies.
 */
export function addJasmineCustomFloatEqualityTester(tolerance = 5e-8) {
  jasmine.addCustomEqualityTester((a, b) => {  // Custom float equality
    if (a === +a && b === +b && (a !== (a | 0) || b !== (b | 0))) {
      return Math.abs(a - b) < tolerance;
    }
    return;
  });
}

/** The minimum interface provided by a test fake. */
export interface MediapipeTasksFake {
  graph: CalculatorGraphConfig|undefined;
  calculatorName: string;
  attachListenerSpies: jasmine.Spy[];
}

/** An map of field paths to values */
export type FieldPathToValue = [string[] | string, unknown];

/**
 * Verifies that the graph has been initialized and that it contains the
 * provided options.
 */
export function verifyGraph(
    tasksFake: MediapipeTasksFake,
    expectedCalculatorOptions?: FieldPathToValue,
    expectedBaseOptions?: FieldPathToValue,
    ): void {
  expect(tasksFake.graph).toBeDefined();
  // Our graphs should have at least one node in them for processing, and
  // sometimes one additional one for keeping alive certain streams in memory.
  expect(tasksFake.graph!.getNodeList().length).toBeGreaterThanOrEqual(1);
  expect(tasksFake.graph!.getNodeList().length).toBeLessThanOrEqual(2);
  const node = tasksFake.graph!.getNodeList()[0].toObject();
  expect(node).toEqual(
      jasmine.objectContaining({calculator: tasksFake.calculatorName}));

  if (expectedBaseOptions) {
    const [fieldPath, value] = expectedBaseOptions;
    let proto = (node.options as {ext: {baseOptions: unknown}}).ext.baseOptions;
    for (const fieldName of (
             Array.isArray(fieldPath) ? fieldPath : [fieldPath])) {
      proto = ((proto ?? {}) as Record<string, unknown>)[fieldName];
    }
    expect(proto).toEqual(value);
  }

  if (expectedCalculatorOptions) {
    const [fieldPath, value] = expectedCalculatorOptions;
    let proto = (node.options as {ext: unknown}).ext;
    for (const fieldName of (
             Array.isArray(fieldPath) ? fieldPath : [fieldPath])) {
      proto = ((proto ?? {}) as Record<string, unknown>)[fieldName];
    }
    expect(proto).toEqual(value);
  }
}

/**
 * Verifies all listeners (as exposed by `.attachListenerSpies`) have been
 * attached at least once since the last call to `verifyListenersRegistered()`.
 * This helps us to ensure that listeners are re-registered with every graph
 * update.
 */
export function verifyListenersRegistered(tasksFake: MediapipeTasksFake): void {
  for (const spy of tasksFake.attachListenerSpies) {
    expect(spy.calls.count()).toBeGreaterThanOrEqual(1);
    spy.calls.reset();
  }
}

/**
 * Copyright 2024 The MediaPipe Authors.
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

import {BaseOptions, TaskRunnerOptions} from '../../../../tasks/web/core/task_runner_options';

/**
 * Options to configure the WebGPU device for LLM Inference task.
 */
export declare interface WebGpuOptions {
  device?: GPUDevice;
  // TODO: b/327685206 - Fill Adapter infor for LLM Web task
  adapterInfo?: GPUAdapterInfo;
}

/**
 * Options to configure the model loading and processing for LLM Inference task.
 */
export declare interface LlmBaseOptions extends BaseOptions {
  gpuOptions?: WebGpuOptions;
}

// TODO: b/324482487 - Support customizing config for Web task of LLM Inference.
/** Options to configure the MediaPipe LLM Inference Task */
export declare interface LlmInferenceOptions extends TaskRunnerOptions {
  /** Options to configure the loading of the model assets. */
  baseOptions?: LlmBaseOptions;
}

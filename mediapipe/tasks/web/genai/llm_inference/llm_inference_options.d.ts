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

import {
  BaseOptions,
  TaskRunnerOptions,
} from '../../../../tasks/web/core/task_runner_options';

/**
 * Options to configure the WebGPU device for LLM Inference task.
 */
export declare interface WebGpuOptions {
  /**
   * The WebGPU device to perform the LLM Inference task.
   * `LlmInference.createWebGpuDevice()` provides the device with
   * performance-prioritized configurations.
   */
  device?: GPUDevice;

  // TODO: b/327685206 - Fill Adapter infor for LLM Web task
  /**
   * The information of WebGPU adapter, which will be used to optimize the
   * performance for LLM Inference task.
   */
  adapterInfo?: GPUAdapterInfo;
}

/**
 * Options to configure the model loading and processing for LLM Inference task.
 */
export declare interface LlmBaseOptions extends BaseOptions {
  gpuOptions?: WebGpuOptions;
}

/** Options to configure the MediaPipe LLM Inference Task */
export declare interface LlmInferenceOptions extends TaskRunnerOptions {
  /** Options to configure the LLM model loading and processing. */
  baseOptions?: LlmBaseOptions;

  /**
   * Maximum number of the combined input and output tokens.
   */
  maxTokens?: number;

  /**
   * The number of candidate tokens to sample from the softmax output in top-k
   * sampling.
   */
  topK?: number;

  /**
   * The temperature used to scale the logits before computing softmax.
   */
  temperature?: number;

  /**
   * Random seed for sampling tokens.
   */
  randomSeed?: number;

  /**
   * The LoRA ranks that will be used during inference.
   */
  loraRanks?: number[];

  /**
   * The number of responses to generate for 'generateResponses' calls,
   * defaulting to 1.
   * In order to ensure variation of responses, you should set topK > 1 and
   * temperature > 0 in the task options; otherwise sampling will collapse to
   * greedy sampling, resulting in all generated responses having the same
   * results.
   * Also, note that increasing this will slow down `generateResponse` calls,
   * because it will still generate many responses, even though it only returns
   * the top result.
   */
  numResponses?: number;

  /**
   * When set to true, will force inference to be performed with F32 precision.
   * Useful for debugging F16 precision-related errors.
   */
  forceF32?: boolean;

  /**
   * When set > 0, will enable vision modality usage. Will also enable streaming
   * loading, and therefore is not compatible with "converted" models.
   */
  maxNumImages?: number;

  /**
   * When set to true, will enable audio modality usage. Will also enable
   * streaming loading, and therefore is not compatible with "converted" models.
   */
  supportAudio?: boolean;
}

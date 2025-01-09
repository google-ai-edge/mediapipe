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

import {FilesetResolver as FilesetResolverImpl} from '../../../tasks/web/core/fileset_resolver';
import {
  LlmInference as LlmInferenceImpl,
  LoraModel as LoraModelImpl,
} from '../../../tasks/web/genai/llm_inference/llm_inference';

// Declare the variables locally so that Rollup in OSS includes them explicitly
// as exports.
const FilesetResolver = FilesetResolverImpl;
const LlmInference = LlmInferenceImpl;
const LoraModel = LoraModelImpl;

export {FilesetResolver, LlmInference, LoraModel};

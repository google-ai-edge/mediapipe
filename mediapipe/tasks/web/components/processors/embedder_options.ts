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

import {EmbedderOptions as EmbedderOptionsProto} from '../../../../tasks/cc/components/processors/proto/embedder_options_pb';
import {EmbedderOptions} from '../../../../tasks/web/core/embedder_options';

/**
 * Converts a EmbedderOptions object to its Proto representation, optionally
 * based on existing definition.
 * @param options The options object to convert to a Proto. Only options that
 *     are expliclty provided are set.
 * @param baseOptions A base object that options can be merged into.
 */
export function convertEmbedderOptionsToProto(
    options: EmbedderOptions,
    baseOptions?: EmbedderOptionsProto): EmbedderOptionsProto {
  const embedderOptions =
      baseOptions ? baseOptions.clone() : new EmbedderOptionsProto();

  if (options.l2Normalize !== undefined) {
    embedderOptions.setL2Normalize(options.l2Normalize);
  } else if ('l2Normalize' in options) {  // Check for undefined
    embedderOptions.clearL2Normalize();
  }

  if (options.quantize !== undefined) {
    embedderOptions.setQuantize(options.quantize);
  } else if ('quantize' in options) {  // Check for undefined
    embedderOptions.clearQuantize();
  }

  return embedderOptions;
}

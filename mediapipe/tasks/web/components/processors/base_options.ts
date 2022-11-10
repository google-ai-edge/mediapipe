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

import {InferenceCalculatorOptions} from '../../../../calculators/tensor/inference_calculator_pb';
import {Acceleration} from '../../../../tasks/cc/core/proto/acceleration_pb';
import {BaseOptions as BaseOptionsProto} from '../../../../tasks/cc/core/proto/base_options_pb';
import {ExternalFile} from '../../../../tasks/cc/core/proto/external_file_pb';
import {BaseOptions} from '../../../../tasks/web/core/base_options';

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

/**
 * Converts a BaseOptions API object to its Protobuf representation.
 * @throws If neither a model assset path or buffer is provided
 */
export async function convertBaseOptionsToProto(
    updatedOptions: BaseOptions,
    currentOptions?: BaseOptionsProto): Promise<BaseOptionsProto> {
  const result =
      currentOptions ? currentOptions.clone() : new BaseOptionsProto();

  await configureExternalFile(updatedOptions, result);
  configureAcceleration(updatedOptions, result);

  return result;
}

/**
 * Configues the `externalFile` option and validates that a single model is
 * provided.
 */
async function configureExternalFile(
    options: BaseOptions, proto: BaseOptionsProto) {
  const externalFile = proto.getModelAsset() || new ExternalFile();
  proto.setModelAsset(externalFile);

  if (options.modelAssetPath || options.modelAssetBuffer) {
    if (options.modelAssetPath && options.modelAssetBuffer) {
      throw new Error(
          'Cannot set both baseOptions.modelAssetPath and baseOptions.modelAssetBuffer');
    }

    let modelAssetBuffer = options.modelAssetBuffer;
    if (!modelAssetBuffer) {
      const response = await fetch(options.modelAssetPath!.toString());
      modelAssetBuffer = new Uint8Array(await response.arrayBuffer());
    }
    externalFile.setFileContent(modelAssetBuffer);
  }

  if (!externalFile.hasFileContent()) {
    throw new Error(
        'Either baseOptions.modelAssetPath or baseOptions.modelAssetBuffer must be set');
  }
}

/** Configues the `acceleration` option. */
function configureAcceleration(options: BaseOptions, proto: BaseOptionsProto) {
  const acceleration = proto.getAcceleration() ?? new Acceleration();
  if (options.delegate === 'gpu') {
    acceleration.setGpu(new InferenceCalculatorOptions.Delegate.Gpu());
  } else {
    acceleration.setTflite(new InferenceCalculatorOptions.Delegate.TfLite());
  }
  proto.setAcceleration(acceleration);
}

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

import {BaseOptions as BaseOptionsProto} from '../../../../tasks/cc/core/proto/base_options_pb';
import {ExternalFile} from '../../../../tasks/cc/core/proto/external_file_pb';
import {BaseOptions} from '../../../../tasks/web/core/base_options';

// The OSS JS API does not support the builder pattern.
// tslint:disable:jspb-use-builder-pattern

/**
 * Converts a BaseOptions API object to its Protobuf representation.
 * @throws If neither a model assset path or buffer is provided
 */
export async function convertBaseOptionsToProto(baseOptions: BaseOptions):
    Promise<BaseOptionsProto> {
  if (baseOptions.modelAssetPath && baseOptions.modelAssetBuffer) {
    throw new Error(
        'Cannot set both baseOptions.modelAssetPath and baseOptions.modelAssetBuffer');
  }
  if (!baseOptions.modelAssetPath && !baseOptions.modelAssetBuffer) {
    throw new Error(
        'Either baseOptions.modelAssetPath or baseOptions.modelAssetBuffer must be set');
  }

  let modelAssetBuffer = baseOptions.modelAssetBuffer;
  if (!modelAssetBuffer) {
    const response = await fetch(baseOptions.modelAssetPath!.toString());
    modelAssetBuffer = new Uint8Array(await response.arrayBuffer());
  }

  const proto = new BaseOptionsProto();
  const externalFile = new ExternalFile();
  externalFile.setFileContent(modelAssetBuffer);
  proto.setModelAsset(externalFile);
  return proto;
}

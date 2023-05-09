/* Copyright 2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.google.mediapipe.tasks.core;

import com.google.mediapipe.calculator.proto.InferenceCalculatorProto;
import com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptions;
import com.google.mediapipe.tasks.core.proto.AccelerationProto;
import com.google.mediapipe.tasks.core.proto.BaseOptionsProto;
import com.google.mediapipe.tasks.core.proto.ExternalFileProto;
import com.google.protobuf.ByteString;

/**
 * MediaPipe Tasks options base class. Any MediaPipe task-specific options class should extend
 * {@link TaskOptions}.
 */
public abstract class TaskOptions {
  /**
   * Converts a MediaPipe Tasks task-specific options to a {@link CalculatorOptions} protobuf
   * message.
   */
  public abstract CalculatorOptions convertToCalculatorOptionsProto();

  /**
   * Converts a {@link BaseOptions} instance to a {@link BaseOptionsProto.BaseOptions} protobuf
   * message.
   */
  protected BaseOptionsProto.BaseOptions convertBaseOptionsToProto(BaseOptions options) {
    ExternalFileProto.ExternalFile.Builder externalFileBuilder =
        ExternalFileProto.ExternalFile.newBuilder();
    options.modelAssetPath().ifPresent(externalFileBuilder::setFileName);
    options
        .modelAssetFileDescriptor()
        .ifPresent(
            fd ->
                externalFileBuilder.setFileDescriptorMeta(
                    ExternalFileProto.FileDescriptorMeta.newBuilder().setFd(fd).build()));
    options
        .modelAssetBuffer()
        .ifPresent(
            modelBuffer -> {
              modelBuffer.rewind();
              externalFileBuilder.setFileContent(ByteString.copyFrom(modelBuffer));
            });
    AccelerationProto.Acceleration.Builder accelerationBuilder =
        AccelerationProto.Acceleration.newBuilder();
    switch (options.delegate()) {
      case CPU:
        accelerationBuilder.setTflite(
            InferenceCalculatorProto.InferenceCalculatorOptions.Delegate.TfLite
                .getDefaultInstance());
        break;
      case GPU:
        accelerationBuilder.setGpu(
            InferenceCalculatorProto.InferenceCalculatorOptions.Delegate.Gpu.newBuilder()
                .setUseAdvancedGpuApi(true)
                .build());
        break;
    }
    return BaseOptionsProto.BaseOptions.newBuilder()
        .setModelAsset(externalFileBuilder.build())
        .setAcceleration(accelerationBuilder.build())
        .build();
  }
}

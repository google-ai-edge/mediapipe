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

import com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptions;
import com.google.mediapipe.tasks.core.proto.BaseOptionsProto;
import com.google.protobuf.Any;

/**
 * MediaPipe Tasks options base class. Any MediaPipe task-specific options class should extend
 * {@link TaskOptions} and implement exactly one of convertTo*Proto() methods.
 */
public abstract class TaskOptions {
  /**
   * Converts a MediaPipe Tasks task-specific options to a {@link CalculatorOptions} protobuf
   * message.
   */
  public CalculatorOptions convertToCalculatorOptionsProto() {
    return null;
  }

  /** Converts a MediaPipe Tasks task-specific options to an proto3 {@link Any} message. */
  public Any convertToAnyProto() {
    return null;
  }

  /**
   * Converts a {@link BaseOptions} instance to a {@link BaseOptionsProto.BaseOptions} protobuf
   * message.
   */
  protected BaseOptionsProto.BaseOptions convertBaseOptionsToProto(BaseOptions options) {
    return BaseOptionsUtils.convertBaseOptionsToProto(options);
  }
}

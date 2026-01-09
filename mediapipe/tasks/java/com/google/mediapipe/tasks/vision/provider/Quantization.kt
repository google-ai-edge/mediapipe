// Copyright 2026 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.mediapipe.tasks.vision.provider

/** The quantization type of the model. */
enum class Quantization(val shortName: String) {
  /** 16-bit float quantization. */
  FLOAT16("fp16"),

  /** 32-bit float quantization. */
  FLOAT32("fp32"),

  /** 8-bit integer quantization. */
  INT8("int8");

  companion object {
    /**
     * Factory method to create a [Quantization] from the model metadata.
     *
     * @param shortName a string representing the quantization type.
     * @return the [Quantization] enum value.
     * @throws IllegalArgumentException if the description is not a supported quantization type.
     */
    fun fromCanonicalName(shortName: String): Quantization {
      return values().find { it.shortName == shortName }
        ?: throw IllegalArgumentException("Unsupported quantization type: $shortName")
    }
  }
}

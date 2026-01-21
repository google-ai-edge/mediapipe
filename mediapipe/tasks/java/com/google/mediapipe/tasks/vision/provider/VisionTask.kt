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

/** Represents a single vision task configuration. */
abstract class VisionTask {
  /** The name of the task (e.g., "FaceDetector"). */
  abstract val name: String

  /**
   * Gets the type of the vision task, derived from the task name (e.g., "FaceDetector").
   *
   * @return The enum constant for this task.
   * @throws IllegalArgumentException if the name does not correspond to a valid task.
   */
  val type: Type by lazy {
    // Convert PascalCase name to UPPER_SNAKE_CASE for enum matching
    val enumName = name.replace(Regex("(?<=[a-z])(?=[A-Z])"), "_").uppercase()
    try {
      Type.valueOf(enumName)
    } catch (e: IllegalArgumentException) {
      throw IllegalArgumentException(
        "Invalid task name '$name'. No matching BaseTask found for '$enumName'. " +
          "Available tasks are: ${Type.values().joinToString { it.name }}"
      )
    }
  }

  /** Defines the types of MediaPipe vision tasks and their corresponding model file extensions. */
  enum class Type(
    /** The required model file extension for the vision task ("task" or "tflite"). */
    val modelExtension: String
  ) {
    FACE_DETECTOR("tflite"),
    FACE_LANDMARKER("task"),
    GESTURE_RECOGNIZER("task"),
    HAND_LANDMARKER("task"),
    IMAGE_CLASSIFIER("tflite"),
    IMAGE_EMBEDDER("tflite"),
    IMAGE_SEGMENTER("tflite"),
    INTERACTIVE_SEGMENTER("tflite"),
    OBJECT_DETECTOR("tflite"),
    POSE_LANDMARKER("task"),
  }

  abstract val defaultModel: String
  abstract val models: List<String>
}

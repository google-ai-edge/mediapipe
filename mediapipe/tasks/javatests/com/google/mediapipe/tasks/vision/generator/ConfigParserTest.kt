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

package com.google.mediapipe.tasks.vision.generator

import com.google.common.truth.Truth.assertThat
import com.google.gson.JsonSyntaxException
import java.io.File
import java.io.FileNotFoundException
import kotlin.test.assertFailsWith
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class ConfigParserTest {

  @Rule @JvmField val tempFolder = TemporaryFolder()

  private lateinit var tasksJson: File

  @Before
  fun setUp() {
    tasksJson = tempFolder.newFile("tasks.json")
  }

  @Test
  fun validJsonParsesCorrectly() {
    tasksJson.writeText(VALID_TASKS_JSON)
    val parser = ConfigParser(tempFolder.root)
    val config = parser.parseConfigurationFromJson()

    assertThat(config.tasks).hasSize(2)

    val faceDetectorTask = config.tasks["FaceDetector"]
    assertThat(faceDetectorTask).isNotNull()
    assertThat(faceDetectorTask!!.name).isEqualTo("FaceDetector")
    assertThat(faceDetectorTask.defaultModel).isEqualTo("face_detection_short_range.tflite")
    assertThat(faceDetectorTask.models).containsExactly("face_detection_short_range.tflite")

    val imageClassifierTask = config.tasks["ImageClassifier"]
    assertThat(imageClassifierTask).isNotNull()
    assertThat(imageClassifierTask!!.name).isEqualTo("ImageClassifier")
    assertThat(imageClassifierTask.defaultModel).isEqualTo("efficientnet_lite0.tflite")
    assertThat(imageClassifierTask.models).containsExactly("efficientnet_lite0.tflite")
  }

  @Test
  fun missingTasksJsonThrowsException() {
    tempFolder.delete()
    tempFolder.create()
    val parser = ConfigParser(tempFolder.root)
    assertFailsWith<FileNotFoundException> { parser.parseConfigurationFromJson() }
  }

  @Test
  fun emptyTasksJsonThrowsException() {
    tasksJson.writeText("")
    val parser = ConfigParser(tempFolder.root)
    assertFailsWith<Exception> { parser.parseConfigurationFromJson() }
  }

  @Test
  fun invalidJsonThrowsException() {
    tasksJson.writeText("{invalid json}")
    val parser = ConfigParser(tempFolder.root)
    assertFailsWith<JsonSyntaxException> { parser.parseConfigurationFromJson() }
  }

  @Test
  fun emptyModelsThrowsException() {
    tasksJson.writeText(
      """
            {
              "tasks": {
                "FaceDetector": {
                  "models": []
                }
              }
            }
            """
    )
    val parser = ConfigParser(tempFolder.root)
    val exception = assertFailsWith<RuntimeException> { parser.parseConfigurationFromJson() }
    assertThat(exception.message).startsWith("No models found for task: FaceDetector")
  }

  @Test
  fun missingModelsThrowsException() {
    tasksJson.writeText(
      """
            {
              "tasks": {
                "FaceDetector": {}
              }
            }
            """
    )
    val parser = ConfigParser(tempFolder.root)
    val exception = assertFailsWith<RuntimeException> { parser.parseConfigurationFromJson() }
    assertThat(exception.message).startsWith("No models found for task: FaceDetector")
  }

  companion object {
    private const val VALID_TASKS_JSON =
      """
            {
              "compileSdk": 34,
              "minSdk": 21,
              "npuModuleDelivery": "on-demand",
              "deviceTargetingConfiguration": "devices.json",
              "tasks": {
                "FaceDetector": {
                  "defaultModel": "face_detection_short_range.tflite",
                  "models": [
                    "face_detection_short_range.tflite"
                  ],
                  "delivery": "install-time"
                },
                "ImageClassifier": {
                  "models": [
                    "efficientnet_lite0.tflite"
                  ],
                  "delivery": "on-demand"
                }
              }
            }
            """
  }
}

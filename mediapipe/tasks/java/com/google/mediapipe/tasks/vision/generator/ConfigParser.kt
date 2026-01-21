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

import com.google.gson.Gson
import com.google.mediapipe.tasks.vision.provider.VisionTask
import com.google.mediapipe.tasks.vision.provider.VisionTaskImpl
import java.io.File
import java.io.IOException

/**
 * Parses the tasks.json configuration file.
 *
 * @property rootDir The root directory of the project, used to find tasks.json.
 */
class ConfigParser(private val rootDir: File) {

  companion object {
    /** The name of the JSON configuration file. */
    private const val JSON_FILE = "tasks.json"
  }

  /**
   * Parses a JSON configuration file to create a VisionProviderConfig.
   *
   * @return The parsed [VisionProviderConfig].
   * @throws IOException if the configuration file cannot be read.
   */
  @Throws(IOException::class)
  fun parseConfigurationFromJson(): VisionProviderConfig {
    val tasksMap = mutableMapOf<String, VisionTask>()
    val configFile = rootDir.resolve(JSON_FILE)
    val jsonContent = configFile.readText()

    val gson = Gson()
    val parsedWrapper = gson.fromJson(jsonContent, TasksWrapper::class.java)
    if (parsedWrapper == null) {
      throw RuntimeException("Failed to parse JSON configuration file.")
    }
    if (parsedWrapper.tasks == null) {
      throw RuntimeException("No tasks found in JSON configuration file.")
    }
    for ((name, config) in parsedWrapper.tasks) {
      if (config.models == null || config.models.isEmpty()) {
        throw RuntimeException("No models found for task: $name")
      }
      val defaultModel = config.defaultModel ?: config.models.first()
      tasksMap[name] = VisionTaskImpl(name, defaultModel, config.models)
    }
    return VisionProviderConfig(tasksMap)
  }
}

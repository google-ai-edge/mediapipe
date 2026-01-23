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

import java.io.File
import java.io.FileInputStream
import java.util.Properties
import org.gradle.api.Plugin
import org.gradle.api.initialization.Settings

/**
 * A Gradle plugin that generates and includes dynamic feature modules for AI Packs and QNN
 * libraries.
 */
class ModuleGeneratorPlugin : Plugin<Settings> {
  /**
   * Applies the plugin to the Gradle [Settings]. This method triggers the generation and inclusion
   * of AI Pack and QNN modules.
   *
   * @param settings The Gradle [Settings] object.
   */
  override fun apply(settings: Settings) {
    checkDeviceTargetingConfigApi(settings)

    val generatedModuleDirs =
      AiPackModuleGenerator().generateAndIncludeModules(settings.rootDir.path)
    generatedModuleDirs.forEach { moduleDir ->
      val moduleName = moduleDir.name
      settings.include(":$moduleName")
      settings.project(":$moduleName").projectDir = moduleDir
      println("✅ Included module: :$moduleName")
    }
  }

  /**
   * Checks if `android.experimental.enableDeviceTargetingConfigApi=true` is set in
   * `gradle.properties`.
   *
   * @param settings The Gradle [Settings] object.
   * @throws IllegalStateException if the property is not set to true.
   */
  private fun checkDeviceTargetingConfigApi(settings: Settings) {
    val gradlePropertiesFile = File(settings.rootDir, "gradle.properties")
    if (!gradlePropertiesFile.exists()) {
      throw IllegalStateException(
        "gradle.properties file not found in root project directory. " +
          "Please ensure 'android.experimental.enableDeviceTargetingConfigApi=true' is set."
      )
    }
    val properties = Properties()
    FileInputStream(gradlePropertiesFile).use { input -> properties.load(input) }
    val propertyValue =
      properties.getProperty("android.experimental.enableDeviceTargetingConfigApi")
    if (propertyValue != "true") {
      throw IllegalStateException(
        "The property 'android.experimental.enableDeviceTargetingConfigApi' must be set to 'true' in gradle.properties " +
          "to use dynamic feature modules for AI Packs and QNN libraries."
      )
    }
  }
}

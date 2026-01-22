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
import java.util.logging.Logger

class DeviceTargetingSetup(private val deviceTargetingOutputDir: File) {

  private val logger = Logger.getLogger(GenerateVisionProviderTask::class.java.name)

  /**
   * Sets up the device targeting configuration file.
   *
   * @return The [File] for the device targeting configuration.
   */
  internal fun setupDeviceTargeting(): File {
    val destinationFile = deviceTargetingOutputDir.resolve("device_targeting_configuration.xml")
    val resourcePath = "xml/device_targeting_configuration.xml"
    try {
      javaClass.getResourceAsStream(resourcePath).use { inputStream ->
        if (inputStream == null) {
          logger.warning("⚠️ Device targeting configuration not found at path: $resourcePath")
        } else {
          destinationFile.parentFile?.mkdirs()
          destinationFile.outputStream().use { outputStream -> inputStream.copyTo(outputStream) }
          logger.info("✅ Copied plugin resource to ${destinationFile.path}")
        }
      }
    } catch (e: Exception) {
      throw RuntimeException("Failed to copy device targeting configuration", e)
    }
    return destinationFile
  }
}

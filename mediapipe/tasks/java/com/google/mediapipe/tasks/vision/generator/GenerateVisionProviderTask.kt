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
import java.io.IOException
import java.util.logging.Logger

/**
 * Orchestrates the generation of the VisionProvider class.
 *
 * @property rootDir The root directory of the project, used to find tasks.json.
 * @property outputDir The directory where the generated VisionProvider.kt file will be written.
 */
class GenerateVisionProviderTask(private val rootDir: File, private val outputDir: File) {

  private val logger = Logger.getLogger(GenerateVisionProviderTask::class.java.name)

  /**
   * Executes the task to generate the VisionProvider.kt file. This involves parsing the
   * `tasks.json` file, generating Kotlin code, and writing it to a file.
   *
   * @throws IOException if there's an error reading or writing files.
   */
  @Throws(IOException::class)
  fun generateVisionProviderFile() {
    val configParser = ConfigParser(rootDir)
    val config = configParser.parseConfigurationFromJson()

    val codeGenerator = VisionProviderCodeGenerator(config)
    val fileSpec = codeGenerator.generateVisionProviderFile()

    outputDir.mkdirs()
    fileSpec.writeTo(outputDir)
    logger.info("✅ Generated VisionProvider and saved to " + outputDir)
  }
}

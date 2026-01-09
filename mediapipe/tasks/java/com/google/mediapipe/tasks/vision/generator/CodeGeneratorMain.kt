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
import java.nio.file.Paths

/**
 * A command-line tool to invoke the MediaPipe vision provider code generator.
 *
 * This tool parses a `tasks.json` file to configure vision tasks, and then generates th
 * `VisionProvider.kt` source file in the specified output directory.
 *
 * Usage:
 * ```
 * CodeGeneratorMain <path/to/tasks.json> </path/to/output/dir>
 * ```
 */
fun main(args: Array<String>) {
  val filteredArgs = args.filter { !it.startsWith("--") }

  require(filteredArgs.size == 2) {
    "Usage: CodeGeneratorMain <path/to/tasks.json> </path/to/output/dir>"
  }

  val tasksJson = File(filteredArgs[0]).absoluteFile
  val outputDirectory = Paths.get(filteredArgs[1])

  val visionProviderConfig = ConfigParser(tasksJson.parentFile!!).parseConfigurationFromJson()

  println("Writing to directory: $outputDirectory")
  VisionProviderCodeGenerator(visionProviderConfig)
    .generateVisionProviderFile()
    .writeTo(outputDirectory)
  println("Generation complete.")
}

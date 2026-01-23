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
import com.google.mediapipe.tasks.vision.provider.VisionModel
import com.google.mediapipe.tasks.vision.provider.VisionTask
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URI
import java.net.URISyntaxException
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.nio.file.StandardOpenOption
import java.util.logging.Logger

/**
 * Handles the dynamic generation and inclusion of AI Pack modules. This generator creates a
 * separate, self-contained module for each model variant and for each QNN NPU variant.
 */
class AiPackModuleGenerator {

  private val logger = Logger.getLogger(AiPackModuleGenerator::class.java.name)

  companion object {
    /** The default name for the application module. Only used as fallback. */
    private const val DEFAULT_APP_MODULE_NAME = "app"

    /** The list of supported ABIs for all Android modules. */
    val SUPPORTED_ABIS = listOf("arm64-v8a", "armeabi-v7a", "x86", "x86_64")

    /** Common libraries required by most/all QNN HTP/DSP variants. */
    val QNN_COMMON_FILES =
      listOf("libLiteRtDispatch_Qualcomm.so", "libQnnHtp.so", "libQnnSystem.so")

    /** Maps QNN architecture variants to their required library files. */
    val QNN_VARIANT_MAP =
      mapOf(
        "qnn_v68" to listOf("libQnnHtpV68Skel.so", "libQnnHtpV68Stub.so"),
        "qnn_v69" to listOf("libQnnHtpV69Skel.so", "libQnnHtpV69Stub.so"),
        "qnn_v73" to listOf("libQnnHtpV73Skel.so", "libQnnHtpV73Stub.so"),
        "qnn_v75" to listOf("libQnnHtpV75Skel.so", "libQnnHtpV75Stub.so"),
        "qnn_v79" to listOf("libQnnHtpV79Skel.so", "libQnnHtpV79Stub.so"),
      )

    /** Provides a reverse mapping from QNN architecture to a list of compatible SoCs. */
    val QNN_REVERSE_MAP =
      mapOf(
        "qnn_v79" to listOf("sm8750", "sxr2330p"),
        "qnn_v75" to listOf("sm8650"),
        "qnn_v73" to listOf("sm8550", "ssg2115p", "ssg2125p", "sxr1230p"),
        "qnn_v69" to listOf("sm8450", "sm8475", "sxr2230p"),
        "qnn_v68" to listOf("sa8295"),
      )
  }

  private fun getDeliveryMode(delivery: String?): String {
    if (delivery == null) {
      return "<dist:on-demand />"
    }
    return when (delivery.lowercase()) {
      "install-time" -> "<dist:install-time />"
      "fast-follow" -> "<dist:fast-follow />"
      "on-demand" -> "<dist:on-demand />"
      else ->
        throw IllegalArgumentException(
          "Invalid npuModuleDelivery value '$delivery'. Must be 'install-time', 'fast-follow', or 'on-demand'."
        )
    }
  }

  private fun findAppModuleName(rootDir: File): String {
    val moduleDirs =
      rootDir.listFiles { file ->
        file.isDirectory &&
          !file.name.startsWith(".") &&
          file.name != "build" &&
          file.name != "gradle"
      } ?: emptyArray()
    for (moduleDir in moduleDirs) {
      val buildGradle = File(moduleDir, "build.gradle")
      val buildGradleKts = File(moduleDir, "build.gradle.kts")
      try {
        if (buildGradle.exists() && buildGradle.readText().contains("com.android.application")) {
          return moduleDir.name
        }
        if (
          buildGradleKts.exists() && buildGradleKts.readText().contains("com.android.application")
        ) {
          return moduleDir.name
        }
      } catch (e: IOException) {
        logger.warning("⚠️: Failed to read build file in ${moduleDir.path}: ${e.message}")
      }
    }
    logger.warning(
      "⚠️: Could not determine app module name, falling back to '${DEFAULT_APP_MODULE_NAME}'."
    )
    return DEFAULT_APP_MODULE_NAME
  }

  /**
   * Creates the directory structure for a QNN dynamic feature module.
   *
   * @param moduleDir The root directory of the QNN module.
   */
  private fun createQnnModuleStructure(moduleDir: File) {
    val mainSrcDir = File(moduleDir, "src/main").apply { mkdirs() }
    val jniLibsDir = File(mainSrcDir, "jniLibs").apply { mkdirs() }

    for (abi in SUPPORTED_ABIS) {
      val abiDir = File(jniLibsDir, abi).apply { mkdirs() }
      //  // If this is NOT the arm64-v8a directory, create a dummy file. The Android App Bundle
      // toolchain (specifically, bundletool) requires that if a `jniLibs/<abi>` directory exists
      // for one ABI, it must exist for all supported ABIs declared in the build, even if empty.
      // Without this, bundletool will fail to build the app bundle, considering the module invalid.
      // The arm64-v8a directory will contain the actual .so files and does not need a placeholder
      // file.
      if (abi != "arm64-v8a") {
        try {
          File(abiDir, "placeholder.so").createNewFile()
        } catch (e: IOException) {
          logger.warning("Failed to create placeholder file for $abi: ${e.message}")
        }
      }
    }
  }

  /**
   * Generates and writes the `build.gradle.kts` file for a QNN dynamic feature module.
   *
   * @param moduleDir The root directory of the QNN module.
   * @param moduleName The name of the QNN module.
   * @param parsedWrapper The [TasksWrapper] containing configuration details.
   * @param rootProjectName The name of the Gradle root project.
   */
  private fun generateQnnBuildGradle(
    moduleDir: File,
    moduleName: String,
    parsedWrapper: TasksWrapper,
    rootProjectName: String,
  ) {
    val buildFile = File(moduleDir, "build.gradle.kts")
    val namespace = "com.google.mediapipe.provider.qnnwrapper.$moduleName"
    buildFile.writeText(
      """
      plugins {
          id("com.android.dynamic-feature")
          kotlin("android")
      }
      android {
          namespace = "$namespace"
          compileSdk = ${parsedWrapper.compileSdk}
          defaultConfig {
              minSdk = ${parsedWrapper.minSdk}
          }
      }
      dependencies {
          implementation(project(":$rootProjectName"))
      }
      """
        .trimIndent()
    )
  }

  /**
   * Generates and writes the `AndroidManifest.xml` file for a QNN dynamic feature module.
   *
   * @param moduleDir The root directory of the QNN module.
   * @param moduleName The name of the QNN module.
   * @param socList A list of compatible SoCs for this QNN module.
   * @param deliveryModeXml The XML snippet for the delivery mode.
   */
  private fun generateQnnAndroidManifest(
    moduleDir: File,
    moduleName: String,
    socList: List<String>,
    deliveryModeXml: String,
  ) {
    val manifestFile = File(moduleDir, "src/main/AndroidManifest.xml")
    val deviceGroupsXml =
      socList.joinToString(separator = "\n                          ") { soc ->
        """<dist:device-group dist:name="qualcomm_${soc.lowercase()}"/>"""
      }

    manifestFile.writeText(
      """
      <manifest xmlns:android="http://schemas.android.com/apk/res/android"
          xmlns:dist="http://schemas.android.com/apk/distribution">
          <dist:module dist:title="@string/title_qnn_wrapper">
              <dist:delivery>$deliveryModeXml
              </dist:delivery>
              <dist:fusing dist:include="false" />
              <dist:conditions>
                 <dist:device-groups>
                    $deviceGroupsXml
                 </dist:device-groups>
              </dist:conditions>
          </dist:module>
      </manifest>
      """
        .trimIndent()
    )
  }

  /**
   * Generates dynamic feature module files for different QNN variants. It iterates through
   * [QNN_REVERSE_MAP] and creates a Gradle dynamic feature module for each QNN architecture,
   * configuring device targeting based on compatible SoCs.
   *
   * @param rootDir The root directory of the project.
   * @param parsedWrapper The [TasksWrapper] containing configuration details.
   * @return A list of [File] objects, each representing a generated QNN module directory.
   */
  fun generateAndIncludeQnnModules(rootDir: String, parsedWrapper: TasksWrapper): List<File> {
    val generatedModuleDirs = mutableListOf<File>()
    val deliveryModeXml = getDeliveryMode(parsedWrapper.npuModuleDelivery)
    val npuModulesDir = File(rootDir, "build/generated/npu-modules")
    val rootProjectName = findAppModuleName(File(rootDir))

    for ((moduleName, socList) in QNN_REVERSE_MAP) {
      val moduleDir = File(npuModulesDir, moduleName)
      createQnnModuleStructure(moduleDir)
      generateQnnBuildGradle(moduleDir, moduleName, parsedWrapper, rootProjectName)
      generateQnnAndroidManifest(moduleDir, moduleName, socList, deliveryModeXml)
      generatedModuleDirs.add(moduleDir)
      logger.info("✅ Generated QNN module files for: :$moduleName")
    }
    return generatedModuleDirs
  }

  /**
   * The main entry point for generating AI Pack and QNN module files. This function parses the
   * `tasks.json` configuration and generates the necessary module directories and files.
   *
   * @param rootDir The root directory of the project.
   * @return A list of [File] objects, each representing a generated module directory.
   */
  fun generateAndIncludeModules(rootDir: String): List<File> {
    val generatedModuleDirs = mutableListOf<File>()
    val configFile = File(rootDir, "tasks.json")
    val aiPackModulesDir = File(rootDir, "build/generated/aipack-modules").apply { mkdirs() }

    if (!configFile.exists()) {
      logger.warning("⚠️: Vision config file not found at ${configFile.path}")
      return generatedModuleDirs
    }

    try {
      logger.info("✅ Generating AI Pack modules from: ${configFile.name}")
      val jsonContent = configFile.readText()
      val gson = Gson()
      val parsedWrapper = gson.fromJson(jsonContent, TasksWrapper::class.java)
      generateModuleFiles(parsedWrapper.tasks, aiPackModulesDir)
      generatedModuleDirs.addAll(generateAndIncludeQnnModules(rootDir, parsedWrapper))
    } catch (e: IOException) {
      throw RuntimeException("Failed to generate AI Pack modules from ${configFile.path}", e)
    }

    // Add all the new AIPack modules to the list
    aiPackModulesDir
      .listFiles { file -> file.isDirectory }
      ?.forEach { moduleDir ->
        generatedModuleDirs.add(moduleDir)
        logger.info("✅ Generated module files for: :${moduleDir.name}")
      }
    return generatedModuleDirs
  }

  /**
   * Generates the build.gradle.kts and asset files for each AI Pack module. This function iterates
   * through the provided tasks and models, downloads the model files, and sets up the necessary
   * Gradle build scripts for each AI Pack.
   *
   * @param tasks A map of task names to their [VisionTaskConfig].
   * @param aiPackModulesDir The directory where AI Pack modules will be generated.
   * @throws IOException if file operations fail.
   */
  @Throws(IOException::class)
  private fun generateModuleFiles(tasks: Map<String, VisionTaskConfig>, aiPackModulesDir: File) {
    for ((taskName, config) in tasks) {
      for (modelName in config.models) {
        val visionModel =
          VisionModel.fromCanonicalName(VisionTask.Type.fromName(taskName), modelName)
        val modelUrls = visionModel.createModelUrls()
        val variants = visionModel.getVariants()
        val moduleDirName = "aipack_${modelName}"
        val moduleDir = File(aiPackModulesDir, moduleDirName)

        // Create a directory for each model variant and download the model file.
        for (i in variants.indices) {
          val assetsDir =
            File(moduleDir, "src/main/assets/model#group_" + variants[i]).apply { mkdirs() }
          val modelFileName = visionModel.modelName
          val destinationFile = File(assetsDir, modelFileName + visionModel.task.modelExtension)
          downloadFile(modelUrls[i], destinationFile)
        }

        val buildScriptPath = moduleDir.toPath().resolve("build.gradle.kts")
        val buildScriptContent = generateAIPackBuildScript(moduleDirName, config.delivery)
        Files.write(
          buildScriptPath,
          buildScriptContent.toByteArray(StandardCharsets.UTF_8),
          StandardOpenOption.CREATE,
          StandardOpenOption.TRUNCATE_EXISTING,
        )
      }
    }
  }

  /**
   * Downloads a file from the given [sourceUrl] to the [destinationFile]. If the [destinationFile]
   * already exists, the download is skipped.
   *
   * @param sourceUrl The URL of the file to download.
   * @param destinationFile The [File] to save the downloaded content to.
   */
  private fun downloadFile(sourceUrl: String, destinationFile: File) {
    if (destinationFile.exists()) {
      return
    }
    logger.fine("⬇️ Downloading $sourceUrl...")
    try {
      val uri = URI(sourceUrl)
      val connection = uri.toURL().openConnection() as HttpURLConnection
      try {
        if (connection.responseCode == HttpURLConnection.HTTP_OK) {
          connection.inputStream.use { input ->
            Files.copy(input, destinationFile.toPath(), StandardCopyOption.REPLACE_EXISTING)
          }
        } else {
          logger.severe(
            "❌ Failed to download $sourceUrl. " +
              "Server responded with code: ${connection.responseCode}"
          )
        }
      } finally {
        connection.disconnect()
      }
    } catch (e: URISyntaxException) {
      logger.severe("❌ Invalid URL syntax for $sourceUrl: ${e.message}")
    } catch (e: IOException) {
      logger.severe("❌ Error downloading $sourceUrl: ${e.message}")
    }
  }

  /**
   * Generates the content of a `build.gradle.kts` file for an AI Pack module. The generated script
   * applies the "com.android.ai-pack" plugin and configures the pack name and delivery type.
   *
   * @param packName The name of the AI Pack.
   * @return The content of the `build.gradle.kts` file as a String.
   */
  private fun generateAIPackBuildScript(packName: String, deliveryType: String?): String {
    val delivery = deliveryType ?: "install-time"
    return """
        plugins {
            id("com.android.ai-pack")
        }
        aiPack {
            packName = "$packName"
            dynamicDelivery {
                deliveryType = "$delivery"
            }
        }
        """
      .trimIndent()
  }
}

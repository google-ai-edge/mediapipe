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

import com.android.build.api.dsl.ApplicationExtension
import com.android.build.api.dsl.DynamicFeatureExtension
import java.io.File
import java.io.IOException
import org.gradle.api.DefaultTask
import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.tasks.TaskProvider
import org.gradle.kotlin.dsl.getByType
import org.gradle.kotlin.dsl.register

/**
 * A Gradle plugin that generates the VisionProvider.kt file and configures the Android extensions
 * for device targeting.
 *
 * ### Plugin Operation
 *
 * This plugin performs two main functions:
 * 1. **Generates `VisionProvider.kt`**: This file is generated during Gradle's configuration phase.
 * 2. **Manages QNN Native Libraries**: It sets up Gradle tasks to extract and package QNN `.so`
 *    libraries for NPU acceleration.
 *
 * #### VisionProvider.kt Generation
 *
 * `VisionProvider.kt` generation occurs directly during the plugin's `apply` method invocation
 * (i.e., during Gradle's configuration phase), not as a separate Gradle task. The
 * `generateVisionProvider` method calls `GenerateVisionProviderTask` to create the
 * `VisionProvider.kt` file in `build/generated/source/main/kotlin`. This directory is expected to
 * be part of the application's main source set, ensuring the generated file is automatically
 * included in the compilation process by tasks such as `compileDebugKotlin` or
 * `compileReleaseKotlin`.
 *
 * #### QNN Library Extraction Workflow
 *
 * The plugin automates the process of fetching and extracting Qualcomm QNN native libraries (`.so`
 * files) from an AAR dependency and packaging them into the correct NPU-specific dynamic feature
 * modules. This ensures that the libraries are available before the Android Gradle Plugin (AGP)
 * tries to merge JNI libraries during the build process.
 *
 * The task dependency flow is as follows:
 * 1. **`setupCreateQnnFolders`**
 *     * **Description**: Creates initial `jniLibs` directories in the root project, including dummy
 *       `CreateFolder.so` files to ensure AGP recognizes the directories.
 *     * **Triggered**: As a dependency of `fetchAndExtractQnnLibs`.
 * 2. **`:<module>:fetchAndExtractQnnLibs`** (e.g., `:qnn-htp:fetchAndExtractQnnLibs`)
 *     * **Description**: For each QNN module (a dynamic feature subproject), this task downloads
 *       the QNN AAR, extracts the relevant `.so` libraries for that module, and places them into
 *       the module's `build/generated/npu-modules/<module>/src/main/jniLibs` directory.
 *     * **Triggered**: This plugin configures AGP's `merge...` tasks to depend on it.
 *     * **Depends On**: `setupCreateQnnFolders`.
 * 3. **AGP `merge[BuildVariant]JniLibFolders` / `merge[BuildVariant]JniLibs`**
 *     * **Description**: Standard AGP tasks responsible for collecting all JNI libraries for a
 *       given build variant (e.g., `mergeDebugJniLibFolders`).
 *     * **Configuration**: This plugin configures these tasks (in both root and subprojects) to
 *       depend on the `fetchAndExtractQnnLibs` task(s), ensuring extraction completes before JNI
 *       library merging begins.
 */
class VisionProviderPlugin : Plugin<Project> {

  override fun apply(project: Project) {
    try {
      val deviceTargetingConfigFile = setupDeviceTargeting(project)
      configureAndroidProject(project, deviceTargetingConfigFile)
      generateVisionProvider(project)
      setupGradleTasks(project)
    } catch (e: IOException) {
      throw RuntimeException(e)
    }
  }

  /**
   * Sets up the device targeting configuration file and generates the VisionProvider.kt file.
   *
   * @param project The Gradle [Project].
   * @return The [File] for the device targeting configuration.
   */
  private fun setupDeviceTargeting(project: Project): File {
    val deviceTargetingOutputDir =
      project.layout.buildDirectory.dir("generated/res/xml").get().asFile
    val deviceTargetingSetup = DeviceTargetingSetup(deviceTargetingOutputDir)
    return deviceTargetingSetup.setupDeviceTargeting()
  }

  /** Generates the VisionProvider entry point for the user. */
  private fun generateVisionProvider(project: Project): File {
    val outputDir = project.layout.buildDirectory.dir("generated/source/main/kotlin").get().asFile
    val generatorTask = GenerateVisionProviderTask(project.rootDir, outputDir)
    generatorTask.generateVisionProviderFile()
  }

  /** Configures Android extensions (dynamic features, bundle, asset packs) */
  private fun configureAndroidProject(project: Project, deviceTargetingConfigFile: File) {

    val androidConfigurator = AndroidConfigurationParser(project.rootDir)
    val androidConfig = androidConfigurator.configureAndroidExtensions(deviceTargetingConfigFile)

    val androidExtension = project.extensions.findByType(ApplicationExtension::class.java)
    if (androidExtension == null) {
      log("⚠️ Warning: Android Application plugin not found on project ${project.name}")
      return
    }

    androidExtension.dynamicFeatures.addAll(androidConfig.dynamicFeatures)
    log("✅ Configured dynamic features: ${androidConfig.dynamicFeatures.joinToString()}")
    androidExtension.assetPacks.addAll(androidConfig.assetPacks)
    log("✅ Configured asset packs: ${androidConfig.assetPacks.joinToString()}")
    androidExtension.bundle {
      deviceTargetingConfig.set(deviceTargetingConfigFile)
      deviceGroup {
        enableSplit = true
        defaultGroup = "other"
      }
      abi { enableSplit = true }
    }
    log("✅ Configured Android App Bundle for device targeting.")
  }

  /**
   * Sets up Gradle tasks (e.g. for extracting QNN libraries for each supported variant).
   *
   * @param project The Gradle [Project].
   */
  private fun setupGradleTasks(project: Project) {
    val setupCreateFolderTask = registerSetupCreateQnnFoldersTask(project)
    val allTasks = registerFetchAndExtractQnnLibsTasks(project, setupCreateFolderTask)
    configureMergeJniLibsTasks(project, allTasks)
    project.logger.lifecycle("✅ Set up rules to extract NPU libraries.")
  }

  private fun registerSetupCreateQnnFoldersTask(project: Project): TaskProvider<DefaultTask> {
    val rootJniLibsFile =
      project.layout.buildDirectory
        .dir("generated/npu-modules/qnn_wrapper/src/main/jniLibs")
        .get()
        .asFile
    return project.tasks.register<DefaultTask>("setupCreateQnnFolders") {
      group = "build"
      description = "Creates folders and files for QNN modules."
      outputs.dir(rootJniLibsFile)
      doLast {
        project.logger.lifecycle("▶️ Setting up folders for NPU delegate...")
        val abis = listOf("armeabi-v7a", "x86", "x86_64")
        abis.forEach { abi ->
          val abiDir = File(rootJniLibsFile, abi)
          abiDir.mkdirs()
          val keepFile = File(abiDir, "CreateFolder.so")
          if (!keepFile.exists()) {
            keepFile.createNewFile()
          }
        }
        project.logger.lifecycle("✅ Created folders in ${rootJniLibsFile.path}")
      }
    }
  }

  private fun registerFetchAndExtractQnnLibsTasks(
    project: Project,
    setupCreateFolderTask: TaskProvider<DefaultTask>,
  ): List<TaskProvider<DefaultTask>> {
    val allTasks = mutableListOf<TaskProvider<DefaultTask>>()
    val qnnVariants = AiPackModuleGenerator.QNN_VARIANT_MAP
    for ((moduleName, _) in qnnVariants) {
      project.rootProject.project(":$moduleName") { subProject ->
        val outputJniLibsDir =
          subProject.layout.buildDirectory.dir("generated/npu-modules/$moduleName/src/main/jniLibs")

        val extractTask =
          subProject.tasks.register<DefaultTask>("fetchAndExtractQnnLibs") {
            group = "build"
            description =
              "Downloads QNN AAR, extracts its native libs, and creates CreateFolder ABI folders."
            val qnnConfig =
              subProject.configurations.detachedConfiguration(
                subProject.dependencies.create(QNN_LIBRARY_DEPENDENCY)
              )
            inputs.files(qnnConfig)
            outputs.dir(outputJniLibsDir)
            this.dependsOn(setupCreateFolderTask)

            doLast {
              val aarFile = qnnConfig.singleFile
              val qnnLibraryExtractor = QnnLibraryExtractor()
              val outputJniLibsFile = outputJniLibsDir.get().asFile
              qnnLibraryExtractor.extractQnnLibraries(aarFile, moduleName, outputJniLibsFile)
            }
          }
        allTasks.add(extractTask)

        // Defer configuration until this subproject is evaluated
        subProject.afterEvaluate {
          val androidExtension = subProject.extensions.getByType<DynamicFeatureExtension>()
          androidExtension.sourceSets.getByName("main").jniLibs.srcDir(outputJniLibsDir)
          subProject.tasks.configureEach { task ->
            if (task.name.startsWith("merge") && task.name.endsWith("JniLibFolders")) {
              task.dependsOn(extractTask)
            }
          }
        }
      }
    }
    return allTasks
  }

  private fun configureMergeJniLibsTasks(
    project: Project,
    allExtractTasks: List<TaskProvider<DefaultTask>>,
  ) {
    project.tasks.configureEach { task ->
      val taskName = task.name
      if (
        taskName == "mergeDebugJniLibFolders" ||
          taskName == "mergeReleaseJniLibFolders" ||
          taskName == "mergeDebugJniLibs" ||
          taskName == "mergeReleaseJniLibs"
      ) {
        task.dependsOn(allExtractTasks)
      }
    }
  }

  companion object {
    private const val QNN_LIBRARY_DEPENDENCY = "com.qualcomm.qti:qnn-runtime:2.39.0@aar"
  }
}

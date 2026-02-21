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

import com.google.mediapipe.tasks.vision.generator.Constants.PACKAGE_NAME
import com.google.mediapipe.tasks.vision.provider.VisionModel
import com.google.mediapipe.tasks.vision.provider.VisionTask
import com.squareup.kotlinpoet.ClassName
import com.squareup.kotlinpoet.CodeBlock
import com.squareup.kotlinpoet.FileSpec
import com.squareup.kotlinpoet.FunSpec
import com.squareup.kotlinpoet.KModifier
import com.squareup.kotlinpoet.MemberName
import com.squareup.kotlinpoet.ParameterSpec
import com.squareup.kotlinpoet.ParameterizedTypeName.Companion.parameterizedBy
import com.squareup.kotlinpoet.PropertySpec
import com.squareup.kotlinpoet.TypeSpec

/** Generates the VisionProvider.kt source file. */
internal class VisionProviderCodeGenerator(private val visionProviderConfig: VisionProviderConfig) {

  companion object {
    private fun createResultListenerTypeName(taskName: String) =
      Constants.MP_IMAGE_CLASS.let { mpImageCn ->
        // InteractiveSegmenter uses ImageSegmenterResult as its result type.
        val resultTaskName = if (taskName == "InteractiveSegmenter") "ImageSegmenter" else taskName
        val resultPackage = "com.google.mediapipe.tasks.vision.${resultTaskName.lowercase()}"
        val resultCn = ClassName(resultPackage, "${resultTaskName}Result")
        Constants.RESULT_LISTENER_BASE_CLASS.parameterizedBy(resultCn, mpImageCn)
      }

    private fun getInternalSettingsParams(taskName: String): List<ParameterSpec> =
      when (taskName) {
        "ImageSegmenter" -> ImageSegmenterSpec.getParams()
        "FaceDetector" -> FaceDetectorSpec.getParams()
        "FaceLandmarker" -> FaceLandmarkerSpec.getParams()
        "HandLandmarker" -> HandLandmarkerSpec.getParams()
        "GestureRecognizer" -> GestureRecognizerSpec.getParams()
        "ImageClassifier" -> ImageClassifierSpec.getParams()
        "ImageEmbedder" -> ImageEmbedderSpec.getParams()
        "InteractiveSegmenter" -> InteractiveSegmenterSpec.getParams()
        "ObjectDetector" -> ObjectDetectorSpec.getParams()
        "PoseLandmarker" -> PoseLandmarkerSpec.getParams()
        else -> throw IllegalArgumentException("Unknown task name: $taskName")
      }
  }

  fun generateVisionProviderFile(): FileSpec {
    val fileBuilder = FileSpec.builder(PACKAGE_NAME, "VisionProvider")
    val visionProviderClassName = ClassName(PACKAGE_NAME, "VisionProvider")
    val contextCn = ClassName("android.content", "Context")
    val futureCn = ClassName("java.util.concurrent", "Future")
    val visionModelCn = ClassName(PACKAGE_NAME, "VisionModel")
    val visionTaskType = ClassName(PACKAGE_NAME, "VisionTask.Type")
    val quantizationCn = ClassName("com.google.mediapipe.tasks.vision.provider", "Quantization")
    val aiPackManagerCn = ClassName("com.google.android.play.core.aipacks", "AiPackManager")
    val aiPackManagerFactoryCn =
      ClassName("com.google.android.play.core.aipacks", "AiPackManagerFactory")
    val splitInstallManagerCn =
      ClassName("com.google.android.play.core.splitinstall", "SplitInstallManager")
    val splitInstallManagerFactoryCn =
      ClassName("com.google.android.play.core.splitinstall", "SplitInstallManagerFactory")

    val classBuilder =
      TypeSpec.classBuilder("VisionProvider")
        .addModifiers(KModifier.PUBLIC)
        .superclass(Constants.VISION_PROVIDER_BASE_CLASS)
        .addFunction(
          FunSpec.constructorBuilder()
            .addParameter("context", contextCn)
            .addParameter(
              ParameterSpec.builder("aiPackManager", aiPackManagerCn)
                .defaultValue("%T.getInstance(context)", aiPackManagerFactoryCn)
                .build()
            )
            .addParameter(
              ParameterSpec.builder("splitInstallManager", splitInstallManagerCn)
                .defaultValue("%T.create(context)", splitInstallManagerFactoryCn)
                .build()
            )
            .callSuperConstructor("context", "aiPackManager", "splitInstallManager")
            .build()
        )
        .addType(
          TypeSpec.companionObjectBuilder()
            .addFunction(
              FunSpec.builder("create")
                .addAnnotation(JvmStatic::class)
                .addParameter(
                  ParameterSpec.builder("context", contextCn)
                    .addAnnotation(Constants.NON_NULL_CLASS)
                    .build()
                )
                .returns(visionProviderClassName)
                .addStatement("return %T(context)", visionProviderClassName)
                .build()
            )
            .build()
        )

    val tasks = visionProviderConfig.tasks.values.toList()
    generateModels(classBuilder, tasks, visionModelCn, quantizationCn, visionTaskType)
    generateSettings(classBuilder, tasks)
    generateCreators(classBuilder, tasks, futureCn)

    fileBuilder.addType(classBuilder.build())
    return fileBuilder.build()
  }

  /**
   * Generates the model enums for each task.
   *
   * Example:
   * ```
   * public enum class ImageSegmenterModel(
   *   override val task: VisionTask.Type,
   *   override val enumName: String,
   *   override val modelName: String,
   *   override val version: String,
   *   override val quantization: Quantization,
   * ) : VisionModel {
   *   SELFIE_SEGMENTATION_V1_FP16(com.google.mediapipe.tasks.vision.provider.VisionTask.Type.IMAGE_SEGMENTER, "selfie_segmentation_v1_fp16", "selfie_segmentation", "v1", Quantization.FLOAT16),
   * }
   * ```
   *
   * @param classBuilder The [TypeSpec.Builder] for the VisionProvider class.
   * @param tasks The list of [VisionTask]s to generate models for.
   * @param visionModelCn The [ClassName] for the VisionModel interface.
   * @param quantizationCn The [ClassName] for the Quantization enum.
   * @param visionTaskTypeCn The [ClassName] for the VisionTask.Type enum.
   */
  private fun generateModels(
    classBuilder: TypeSpec.Builder,
    tasks: Iterable<VisionTask>,
    visionModelCn: ClassName,
    quantizationCn: ClassName,
    visionTaskTypeCn: ClassName,
  ) {
    for (task in tasks) {
      val modelEnumBuilder =
        TypeSpec.enumBuilder("${task.name}Model")
          .addModifiers(KModifier.PUBLIC)
          .addSuperinterface(visionModelCn)
          .primaryConstructor(
            FunSpec.constructorBuilder()
              .addParameter("task", visionTaskTypeCn)
              .addParameter("enumName", Constants.KOTLIN_STRING_CLASS)
              .addParameter("modelName", Constants.KOTLIN_STRING_CLASS)
              .addParameter("version", Constants.KOTLIN_STRING_CLASS)
              .addParameter("quantization", quantizationCn)
              .build()
          )
          .addProperty(
            PropertySpec.builder("task", visionTaskTypeCn, KModifier.OVERRIDE)
              .initializer("task")
              .build()
          )
          .addProperty(
            PropertySpec.builder("enumName", Constants.KOTLIN_STRING_CLASS, KModifier.OVERRIDE)
              .initializer("enumName")
              .build()
          )
          .addProperty(
            PropertySpec.builder("modelName", Constants.KOTLIN_STRING_CLASS, KModifier.OVERRIDE)
              .initializer("modelName")
              .build()
          )
          .addProperty(
            PropertySpec.builder("version", Constants.KOTLIN_STRING_CLASS, KModifier.OVERRIDE)
              .initializer("version")
              .build()
          )
          .addProperty(
            PropertySpec.builder("quantization", quantizationCn, KModifier.OVERRIDE)
              .initializer("quantization")
              .build()
          )

      val taskEnumName = task.name.replace(Regex("(?<=[a-z])(?=[A-Z])"), "_").uppercase()
      val enumConstantMember = MemberName(visionTaskTypeCn, taskEnumName)

      for (modelName in task.models) {
        val model = VisionModel.fromCanonicalName(task.type, modelName)
        val enumConstantName = model.enumName
        val anonymousClassBuilder =
          TypeSpec.anonymousClassBuilder()
            .addSuperclassConstructorParameter(CodeBlock.of(enumConstantMember.toString()))
            .addSuperclassConstructorParameter("%S", enumConstantName)
            .addSuperclassConstructorParameter("%S", model.modelName)
            .addSuperclassConstructorParameter("%S", model.version)
            .addSuperclassConstructorParameter("%T.%L", quantizationCn, model.quantization)
        modelEnumBuilder.addEnumConstant(enumConstantName, anonymousClassBuilder.build())
      }
      classBuilder.addType(modelEnumBuilder.build())
    }
  }

  /**
   * Generates a settings data class.
   *
   * @param classBuilder The [TypeSpec.Builder] for the VisionProvider class.
   * @param settingName The name of the settings class to generate.
   * @param params The parameters for the settings class.
   */
  private fun generateSetting(
    classBuilder: TypeSpec.Builder,
    settingName: String,
    params: List<ParameterSpec>,
  ) {
    val settingsClassName = ClassName("$PACKAGE_NAME.VisionProvider", "${settingName}Settings")
    val dataClassBuilder =
      TypeSpec.classBuilder(settingsClassName).addModifiers(KModifier.PUBLIC, KModifier.DATA)
    val constructorBuilder = FunSpec.constructorBuilder()

    for (param in params) {
      constructorBuilder.addParameter(param)
      dataClassBuilder.addProperty(
        PropertySpec.builder(param.name, param.type).initializer(param.name).build()
      )
      val methodName = "with${param.name.replaceFirstChar { it.uppercase() }}"
      dataClassBuilder.addFunction(
        FunSpec.builder(methodName)
          .addModifiers(KModifier.PUBLIC)
          .returns(settingsClassName)
          .addParameter(param.name, param.type)
          .addStatement("return copy(%N = %N)", param.name, param.name)
          .build()
      )
    }

    dataClassBuilder.primaryConstructor(constructorBuilder.build())
    classBuilder.addType(dataClassBuilder.build())
  }

  /**
   * Generates the settings classes for each task.
   *
   * Example:
   * ```
   * public data class ImageSegmenterSettings(
   *   val outputConfidenceMasks: Boolean = false,
   *   val outputCategoryMask: Boolean = false,
   *   val displayNamesLocale: String = "und",
   *   val runningMode: RunningMode = RunningMode.IMAGE,
   *   val resultListener: ResultListener<ImageSegmenterResult, MPImage> = NoOpResultListener(),
   *   val errorListener: ErrorListener = NoOpErrorListener(),
   * ) {
   *   fun withOutputConfidenceMasks(outputConfidenceMasks: Boolean): ImageSegmenterSettings
   *   fun withOutputCategoryMask(outputCategoryMask: Boolean): ImageSegmenterSettings
   *   fun withDisplayNamesLocale(displayNamesLocale: String): ImageSegmenterSettings
   *   fun withRunningMode(runningMode: RunningMode): ImageSegmenterSettings
   *   fun withResultListener(resultListener: ResultListener<ImageSegmenterResult, MPImage>): ImageSegmenterSettings
   *   fun withErrorListener(errorListener: ErrorListener): ImageSegmenter
   * }
   * ```
   *
   * @param classBuilder The [TypeSpec.Builder] for the VisionProvider class.
   * @param tasks The list of [VisionTask]s to generate settings for.
   */
  private fun generateSettings(classBuilder: TypeSpec.Builder, tasks: Iterable<VisionTask>) {
    generateSetting(classBuilder, "Classifier", ClassifierSpec.getParams())
    for (task in tasks) {
      generateSetting(classBuilder, task.name, getInternalSettingsParams(task.name))
    }
  }

  private fun toClassifierOptions(setting: String): String {
    val params = ClassifierSpec.getParams().joinToString(", ") { "s.${it.name}" }
    return setting +
      "?.let { s -> com.google.mediapipe.tasks.vision.provider.ClassifierSettingsInternal($params) }"
  }

  /**
   * Generates the createX() methods for each task.
   *
   * Example:
   * ```
   * public fun createImageSegmenter(
   *   @NonNull model: com.google.mediapipe.tasks.vision.provider.VisionProvider.ImageSegmenterModel = com.google.mediapipe.tasks.vision.provider.VisionProvider.ImageSegmenterModel.SELFIE_SEGMENTATION_V1_FP16,
   *   @NonNull settings: com.google.mediapipe.tasks.vision.provider.VisionProvider.ImageSegmenterSettings = com.google.mediapipe.tasks.vision.provider.VisionProvider.ImageSegmenterSettings(),
   * ): Future<ImageSegmenter>
   * ```
   *
   * @param classBuilder The [TypeSpec.Builder] for the VisionProvider class.
   * @param tasks The list of [VisionTask]s to generate creators for.
   * @param futureCn The [ClassName] for the Future class.
   */
  private fun generateCreators(
    classBuilder: TypeSpec.Builder,
    tasks: Iterable<VisionTask>,
    futureCn: ClassName,
  ) {
    val classifierSettingsCn =
      ClassName("com.google.mediapipe.tasks.vision.provider.VisionProvider", "ClassifierSettings")

    for (task in tasks) {
      val taskName = task.name
      val modelEnumCn = ClassName("$PACKAGE_NAME.VisionProvider", "${taskName}Model")
      val settingsRecordCn = ClassName("$PACKAGE_NAME.VisionProvider", "${taskName}Settings")
      val internalSettingsCn = ClassName(PACKAGE_NAME, "${taskName}SettingsInternal")
      val taskReturnCn =
        ClassName("com.google.mediapipe.tasks.vision.${taskName.lowercase()}", taskName)
      val futureOfTask = futureCn.parameterizedBy(taskReturnCn)
      val defaultModel = task.defaultModel
      val defaultModelEnumName = VisionModel.fromCanonicalName(task.type, defaultModel).enumName

      val params = getInternalSettingsParams(taskName)
      val settingsParams =
        params
          .map {
            if (it.type == classifierSettingsCn.copy(nullable = true)) {
              "${it.name} = ${toClassifierOptions("settings.${it.name}")}"
            } else {
              "${it.name} = settings.${it.name}"
            }
          }
          .joinToString(", ")

      classBuilder.addFunction(
        FunSpec.builder("create$taskName")
          .addAnnotation(JvmOverloads::class)
          .addModifiers(KModifier.PUBLIC)
          .returns(futureOfTask)
          .addParameter(
            ParameterSpec.builder("model", modelEnumCn)
              .addAnnotation(Constants.NON_NULL_CLASS)
              .defaultValue("%T.%L", modelEnumCn, defaultModelEnumName)
              .build()
          )
          .addParameter(
            ParameterSpec.builder("settings", settingsRecordCn)
              .addAnnotation(Constants.NON_NULL_CLASS)
              .defaultValue("%T()", settingsRecordCn)
              .build()
          )
          .addStatement(
            "return create${taskName}Impl(model, %T($settingsParams))",
            internalSettingsCn,
          )
          .build()
      )
    }
  }
}

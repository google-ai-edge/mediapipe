# Copyright 2022 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Building MediaPipe Tasks AARs."""

load("@build_bazel_rules_android//android:rules.bzl", "android_binary", "android_library")

_CORE_TASKS_JAVA_PROTO_LITE_TARGETS = [
    "//mediapipe/gpu:gpu_origin_java_proto_lite",
    "//mediapipe/tasks/cc/components/containers/proto:classifications_java_proto_lite",
    "//mediapipe/tasks/cc/components/containers/proto:embeddings_java_proto_lite",
    "//mediapipe/tasks/cc/components/containers/proto:landmarks_detection_result_java_proto_lite",
    "//mediapipe/tasks/cc/components/processors/proto:classifier_options_java_proto_lite",
    "//mediapipe/tasks/cc/components/processors/proto:embedder_options_java_proto_lite",
    "//mediapipe/tasks/cc/core/proto:acceleration_java_proto_lite",
    "//mediapipe/tasks/cc/core/proto:base_options_java_proto_lite",
    "//mediapipe/tasks/cc/core/proto:external_file_java_proto_lite",
]

_AUDIO_TASKS_JAVA_PROTO_LITE_TARGETS = [
    "//mediapipe/tasks/cc/audio/audio_classifier/proto:audio_classifier_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/audio/audio_embedder/proto:audio_embedder_graph_options_java_proto_lite",
]

_VISION_TASKS_JAVA_PROTO_LITE_TARGETS = [
    "//mediapipe/tasks/cc/vision/face_detector/proto:face_detector_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/face_geometry/proto:face_geometry_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/face_geometry/proto:face_geometry_java_proto_lite",
    "//mediapipe/tasks/cc/vision/face_geometry/proto:mesh_3d_java_proto_lite",
    "//mediapipe/tasks/cc/vision/face_landmarker/proto:face_blendshapes_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/face_landmarker/proto:face_landmarker_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/face_landmarker/proto:face_landmarks_detector_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/face_stylizer/proto:face_stylizer_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/gesture_recognizer/proto:gesture_classifier_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/gesture_recognizer/proto:gesture_embedder_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/gesture_recognizer/proto:gesture_recognizer_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/gesture_recognizer/proto:hand_gesture_recognizer_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/hand_detector/proto:hand_detector_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarker_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarks_detector_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_roi_refinement_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/holistic_landmarker/proto:holistic_landmarker_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/image_classifier/proto:image_classifier_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/image_embedder/proto:image_embedder_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/image_segmenter/proto:image_segmenter_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/image_segmenter/proto:segmenter_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/object_detector/proto:object_detector_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/pose_detector/proto:pose_detector_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/pose_landmarker/proto:pose_landmarker_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/pose_landmarker/proto:pose_landmarks_detector_graph_options_java_proto_lite",
]

_VISION_TASKS_IMAGE_GENERATOR_JAVA_PROTO_LITE_SRC_TARGETS = [
    "//mediapipe/tasks/cc/vision/image_generator/proto:conditioned_image_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/image_generator/proto:control_plugin_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/image_generator/proto:image_generator_graph_options_java_proto_lite",
]

_VISION_TASKS_IMAGE_GENERATOR_JAVA_PROTO_LITE_TARGETS = [
    "//mediapipe/tasks/cc/vision/face_detector/proto:face_detector_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/face_geometry/proto:face_geometry_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/face_geometry/proto:face_geometry_java_proto_lite",
    "//mediapipe/tasks/cc/vision/face_geometry/proto:mesh_3d_java_proto_lite",
    "//mediapipe/tasks/cc/vision/face_landmarker/proto:face_blendshapes_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/face_landmarker/proto:face_landmarker_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/face_landmarker/proto:face_landmarks_detector_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/image_generator/diffuser:stable_diffusion_iterate_calculator_java_proto_lite",
    "//mediapipe/tasks/cc/vision/image_generator/proto:conditioned_image_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/image_generator/proto:control_plugin_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/image_generator/proto:image_generator_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/image_segmenter/proto:image_segmenter_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/image_segmenter/proto:segmenter_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/image_segmenter/calculators:tensors_to_segmentation_calculator_java_proto_lite",
    "//mediapipe/tasks/cc/vision/hand_detector/proto:hand_detector_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarker_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/vision/hand_landmarker/proto:hand_landmarks_detector_graph_options_java_proto_lite",
]

_TEXT_TASKS_JAVA_PROTO_LITE_TARGETS = [
    "//mediapipe/tasks/cc/text/text_classifier/proto:text_classifier_graph_options_java_proto_lite",
    "//mediapipe/tasks/cc/text/text_embedder/proto:text_embedder_graph_options_java_proto_lite",
]

_GENAI_TASKS_JAVA_PROTO_LITE_TARGETS = [
    "//mediapipe/tasks/java/com/google/mediapipe/tasks/genai/llminference/jni/proto:llm_options_java_proto_lite",
    "//mediapipe/tasks/java/com/google/mediapipe/tasks/genai/llminference/jni/proto:llm_response_context_java_proto_lite",
]

def mediapipe_tasks_core_aar(name, srcs, manifest):
    """Builds medaipipe tasks core AAR.

    Args:
      name: The bazel target name.
      srcs: MediaPipe Tasks' core layer source files.
      manifest: The Android manifest.
    """

    # When "--define ENABLE_TASKS_USAGE_LOGGING=1" is set in the build command,
    # the mediapipe tasks usage logging component will be added into the AAR.
    # This flag is for internal use only.
    native.config_setting(
        name = "enable_tasks_usage_logging",
        define_values = {
            "ENABLE_TASKS_USAGE_LOGGING": "1",
        },
        visibility = ["//visibility:public"],
    )

    mediapipe_tasks_java_proto_srcs = []
    for target in _CORE_TASKS_JAVA_PROTO_LITE_TARGETS:
        mediapipe_tasks_java_proto_srcs.append(
            _mediapipe_tasks_java_proto_src_extractor(target = target),
        )

    for target in _AUDIO_TASKS_JAVA_PROTO_LITE_TARGETS:
        mediapipe_tasks_java_proto_srcs.append(
            _mediapipe_tasks_java_proto_src_extractor(target = target),
        )

    for target in _VISION_TASKS_JAVA_PROTO_LITE_TARGETS:
        mediapipe_tasks_java_proto_srcs.append(
            _mediapipe_tasks_java_proto_src_extractor(target = target),
        )

    for target in _TEXT_TASKS_JAVA_PROTO_LITE_TARGETS:
        mediapipe_tasks_java_proto_srcs.append(
            _mediapipe_tasks_java_proto_src_extractor(target = target),
        )

    for target in _VISION_TASKS_IMAGE_GENERATOR_JAVA_PROTO_LITE_SRC_TARGETS:
        mediapipe_tasks_java_proto_srcs.append(
            _mediapipe_tasks_java_proto_src_extractor(target = target),
        )

    mediapipe_tasks_java_proto_srcs.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/calculators/core:flow_limiter_calculator_java_proto_lite",
        src_out = "com/google/mediapipe/calculator/proto/FlowLimiterCalculatorProto.java",
    ))

    mediapipe_tasks_java_proto_srcs.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/calculators/tensor:inference_calculator_java_proto_lite",
        src_out = "com/google/mediapipe/calculator/proto/InferenceCalculatorProto.java",
    ))

    mediapipe_tasks_java_proto_srcs.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/tasks/cc/vision/image_segmenter/calculators:tensors_to_segmentation_calculator_java_proto_lite",
        src_out = "com/google/mediapipe/tasks/TensorsToSegmentationCalculatorOptionsProto.java",
    ))

    mediapipe_tasks_java_proto_srcs.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/tasks/cc/vision/face_geometry/calculators:geometry_pipeline_calculator_java_proto_lite",
        src_out = "com/google/mediapipe/tasks/vision/facegeometry/calculators/proto/FaceGeometryPipelineCalculatorOptionsProto.java",
    ))

    mediapipe_tasks_java_proto_srcs.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/tasks/cc/vision/image_generator/diffuser:stable_diffusion_iterate_calculator_java_proto_lite",
        src_out = "com/google/mediapipe/calculator/proto/StableDiffusionIterateCalculatorOptionsProto.java",
    ))

    android_library(
        name = name,
        srcs = srcs + [
                   "//mediapipe/java/com/google/mediapipe/framework:java_src",
               ] + mediapipe_java_proto_srcs() +
               mediapipe_tasks_java_proto_srcs +
               select({
                   "//conditions:default": [],
                   "enable_tasks_usage_logging": mediapipe_logging_java_proto_srcs(),
               }),
        javacopts = [
            "-Xep:AndroidJdkLibsChecker:OFF",
        ],
        manifest = manifest,
        deps = [
                   "//mediapipe/calculators/core:flow_limiter_calculator_java_proto_lite",
                   "//mediapipe/calculators/tensor:inference_calculator_java_proto_lite",
                   "//mediapipe/framework:calculator_java_proto_lite",
                   "//mediapipe/framework:calculator_profile_java_proto_lite",
                   "//mediapipe/framework:calculator_options_java_proto_lite",
                   "//mediapipe/framework:mediapipe_options_java_proto_lite",
                   "//mediapipe/framework:packet_factory_java_proto_lite",
                   "//mediapipe/framework:packet_generator_java_proto_lite",
                   "//mediapipe/framework:status_handler_java_proto_lite",
                   "//mediapipe/framework:stream_handler_java_proto_lite",
                   "//mediapipe/framework/formats:classification_java_proto_lite",
                   "//mediapipe/framework/formats:detection_java_proto_lite",
                   "//mediapipe/framework/formats:landmark_java_proto_lite",
                   "//mediapipe/framework/formats:location_data_java_proto_lite",
                   "//mediapipe/framework/formats:rect_java_proto_lite",
                   "//mediapipe/java/com/google/mediapipe/framework:android_framework",
                   "//mediapipe/java/com/google/mediapipe/framework/image",
                   "//mediapipe/tasks/cc/vision/image_segmenter/calculators:tensors_to_segmentation_calculator_java_proto_lite",
                   "//mediapipe/tasks/java/com/google/mediapipe/tasks/core/jni:model_resources_cache_jni",
                   "//third_party:autovalue",
                   "@com_google_protobuf//:protobuf_javalite",
                   "@maven//:androidx_annotation_annotation",
                   "@maven//:com_google_guava_guava",
                   "@maven//:com_google_flogger_flogger",
                   "@maven//:com_google_flogger_flogger_system_backend",
                   "@maven//:com_google_code_findbugs_jsr305",
               ] +
               _AUDIO_TASKS_JAVA_PROTO_LITE_TARGETS +
               _CORE_TASKS_JAVA_PROTO_LITE_TARGETS +
               _TEXT_TASKS_JAVA_PROTO_LITE_TARGETS +
               _VISION_TASKS_JAVA_PROTO_LITE_TARGETS +
               select({
                   "//conditions:default": [],
                   "enable_tasks_usage_logging": [
                       "@maven//:com_google_android_datatransport_transport_api",
                       "@maven//:com_google_android_datatransport_transport_backend_cct",
                       "@maven//:com_google_android_datatransport_transport_runtime",
                   ],
               }),
    )

def mediapipe_tasks_audio_aar(name, srcs, native_library):
    """Builds medaipipe tasks audio AAR.

    Args:
      name: The bazel target name.
      srcs: MediaPipe Audio Tasks' source files.
      native_library: The native library that contains audio tasks' graph and calculators.
    """

    native.genrule(
        name = name + "tasks_manifest_generator",
        outs = ["AndroidManifest.xml"],
        cmd = """
cat > $(OUTS) <<EOF
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.google.mediapipe.tasks.audio">
    <uses-sdk
        android:minSdkVersion="24"
        android:targetSdkVersion="34" />
</manifest>
EOF
""",
    )

    _mediapipe_tasks_aar(
        name = name,
        srcs = srcs,
        manifest = "AndroidManifest.xml",
        java_proto_lite_targets = _CORE_TASKS_JAVA_PROTO_LITE_TARGETS + _AUDIO_TASKS_JAVA_PROTO_LITE_TARGETS,
        native_library = native_library,
    )

def mediapipe_tasks_vision_aar(name, srcs, native_library):
    """Builds medaipipe tasks vision AAR.

    Args:
      name: The bazel target name.
      srcs: MediaPipe Vision Tasks' source files.
      native_library: The native library that contains vision tasks' graph and calculators.
    """

    native.genrule(
        name = name + "tasks_manifest_generator",
        outs = ["AndroidManifest.xml"],
        cmd = """
cat > $(OUTS) <<EOF
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.google.mediapipe.tasks.vision">
    <uses-sdk
        android:minSdkVersion="24"
        android:targetSdkVersion="34" />
</manifest>
EOF
""",
    )

    _mediapipe_tasks_aar(
        name = name,
        srcs = srcs,
        manifest = "AndroidManifest.xml",
        java_proto_lite_targets = _CORE_TASKS_JAVA_PROTO_LITE_TARGETS + _VISION_TASKS_JAVA_PROTO_LITE_TARGETS + [
            "//mediapipe/tasks/cc/vision/image_segmenter/calculators:tensors_to_segmentation_calculator_java_proto_lite",
        ],
        native_library = native_library,
    )

def mediapipe_tasks_vision_image_generator_aar(name, srcs, native_library):
    """Builds medaipipe tasks vision image generator AAR.

    Args:
      name: The bazel target name.
      srcs: MediaPipe Vision Tasks' source files.
      native_library: The native library that contains image generator task's graph and calculators.
    """

    native.genrule(
        name = name + "tasks_manifest_generator",
        outs = ["AndroidManifest.xml"],
        cmd = """
cat > $(OUTS) <<EOF
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.google.mediapipe.tasks.vision.imagegenerator">
    <uses-sdk
        android:minSdkVersion="24"
        android:targetSdkVersion="34" />
</manifest>
EOF
""",
    )
    _mediapipe_tasks_aar(
        name = name,
        srcs = srcs,
        manifest = "AndroidManifest.xml",
        java_proto_lite_targets = _CORE_TASKS_JAVA_PROTO_LITE_TARGETS + _VISION_TASKS_IMAGE_GENERATOR_JAVA_PROTO_LITE_TARGETS,
        native_library = native_library,
    )

def mediapipe_tasks_text_aar(name, srcs, native_library):
    """Builds medaipipe tasks text AAR.

    Args:
      name: The bazel target name.
      srcs: MediaPipe Text Tasks' source files.
      native_library: The native library that contains text tasks' graph and calculators.
    """

    native.genrule(
        name = name + "tasks_manifest_generator",
        outs = ["AndroidManifest.xml"],
        cmd = """
cat > $(OUTS) <<EOF
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.google.mediapipe.tasks.text">
    <uses-sdk
        android:minSdkVersion="24"
        android:targetSdkVersion="34" />
</manifest>
EOF
""",
    )

    _mediapipe_tasks_aar(
        name = name,
        srcs = srcs,
        manifest = "AndroidManifest.xml",
        java_proto_lite_targets = _CORE_TASKS_JAVA_PROTO_LITE_TARGETS + _TEXT_TASKS_JAVA_PROTO_LITE_TARGETS,
        native_library = native_library,
    )

def mediapipe_tasks_genai_aar(name, srcs, native_library):
    """Builds medaipipe tasks text text generator AAR.

    Args:
      name: The bazel target name.
      srcs: MediaPipe Text Generator Tasks' source files.
      native_library: The native library that contains text generator task's graph and calculators.
    """

    native.genrule(
        name = name + "tasks_manifest_generator",
        outs = ["AndroidManifest.xml"],
        cmd = """
cat > $(OUTS) <<EOF
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.google.mediapipe.tasks.genai">
    <uses-sdk
        android:minSdkVersion="24"
        android:targetSdkVersion="34" />
</manifest>
EOF
""",
    )

    mediapipe_genai_java_proto_srcs = []
    mediapipe_genai_java_proto_srcs.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/tasks/java/com/google/mediapipe/tasks/genai/llminference/jni/proto:llm_options_java_proto_lite",
        src_out = "com/google/mediapipe/tasks/genai/llminference/jni/proto/LlmOptionsProto.java",
    ))

    mediapipe_genai_java_proto_srcs.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/tasks/java/com/google/mediapipe/tasks/genai/llminference/jni/proto:llm_response_context_java_proto_lite",
        src_out = "com/google/mediapipe/tasks/genai/llminference/jni/proto/LlmResponseContextProto.java",
    ))

    _mediapipe_tasks_aar(
        name = name,
        srcs = srcs + mediapipe_genai_java_proto_srcs,
        manifest = "AndroidManifest.xml",
        java_proto_lite_targets = _CORE_TASKS_JAVA_PROTO_LITE_TARGETS + _GENAI_TASKS_JAVA_PROTO_LITE_TARGETS,
        native_library = native_library,
    )

def _mediapipe_tasks_aar(name, srcs, manifest, java_proto_lite_targets, native_library):
    """Builds medaipipe tasks AAR."""
    deps = java_proto_lite_targets + [native_library] + [
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/framework/image",
        "//mediapipe/framework:calculator_options_java_proto_lite",
        "//mediapipe/framework:calculator_java_proto_lite",
        "//mediapipe/framework/formats:classification_java_proto_lite",
        "//mediapipe/framework/formats:detection_java_proto_lite",
        "//mediapipe/framework/formats:landmark_java_proto_lite",
        "//mediapipe/framework/formats:location_data_java_proto_lite",
        "//mediapipe/framework/formats:matrix_data_java_proto_lite",
        "//mediapipe/framework/formats:rect_java_proto_lite",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:audiodata",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:detection",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:category",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:classificationresult",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:classifications",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:connection",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:embedding",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:embeddingresult",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:landmark",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:normalizedkeypoint",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/containers:normalized_landmark",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/processors:classifieroptions",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/components/utils:cosinesimilarity",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core:logging",
        "//mediapipe/tasks/java/com/google/mediapipe/tasks/core",
        "//mediapipe/util:color_java_proto_lite",
        "//mediapipe/util:label_map_java_proto_lite",
        "//mediapipe/util:render_data_java_proto_lite",
        "//third_party:autovalue",
        "@maven//:androidx_annotation_annotation",
        "@maven//:com_google_guava_guava",
        "@com_google_protobuf//:protobuf_javalite",
        "//third_party:any_java_proto",
    ]

    deps += select({
        "//conditions:default": ["//third_party:android_jni_opencv_cc_lib"],
        "//mediapipe/framework/port:disable_opencv": [],
        "//third_party:exclude_opencv_so_lib": [],
    })

    android_library(
        name = name + "_android_lib",
        srcs = srcs,
        manifest = manifest,
        proguard_specs = ["//mediapipe/java/com/google/mediapipe/framework:proguard.pgcfg"],
        deps = deps,
    )

    mediapipe_build_aar_with_jni(name, name + "_android_lib")

def _mediapipe_tasks_java_proto_src_extractor(target):
    proto_path = "com/google/" + target.split(":")[0].replace("cc/", "").replace("//", "").replace("_", "") + "/"
    proto_name = target.split(":")[-1].replace("_java_proto_lite", "").replace("_", " ").title().replace(" ", "") + "Proto.java"
    return mediapipe_java_proto_src_extractor(
        target = target,
        src_out = proto_path + proto_name,
    )

def mediapipe_build_aar_with_jni(name, android_library):
    """Builds MediaPipe AAR with jni.

    Args:
      name: The bazel target name.
      android_library: the android library that contains jni.
    """

    # Generates dummy AndroidManifest.xml for dummy apk usage
    # (dummy apk is generated by <name>_dummy_app target below)
    native.genrule(
        name = name + "_binary_manifest_generator",
        outs = [name + "_generated_AndroidManifest.xml"],
        cmd = """
cat > $(OUTS) <<EOF
<manifest
  xmlns:android="http://schemas.android.com/apk/res/android"
  package="dummy.package.for.so">
  <uses-sdk android:minSdkVersion="24"/>
</manifest>
EOF
""",
    )

    # Generates dummy apk including .so files.
    # We extract out .so files and throw away the apk.
    android_binary(
        name = name + "_dummy_app",
        manifest = name + "_generated_AndroidManifest.xml",
        custom_package = "dummy.package.for.so",
        multidex = "native",
        deps = [android_library],
    )

    native.genrule(
        name = name,
        srcs = [android_library + ".aar", name + "_dummy_app_unsigned.apk"],
        outs = [name + ".aar"],
        tags = ["manual"],
        cmd = """
cp $(location {}.aar) $(location :{}.aar)
chmod +w $(location :{}.aar)
origdir=$$PWD
cd $$(mktemp -d)
unzip $$origdir/$(location :{}_dummy_app_unsigned.apk) "lib/*"
find lib -name *_dummy_app.so -delete
cp -r lib jni
zip -r $$origdir/$(location :{}.aar) jni/*/*.so
""".format(android_library, name, name, name, name),
    )

def mediapipe_java_proto_src_extractor(target, src_out, name = ""):
    """Extracts the generated MediaPipe java proto source code from the target.

    Args:
      target: The java proto lite target to be built and extracted.
      src_out: The output java proto src code path.
      name: The optional bazel target name.

    Returns:
      The output java proto src code path.
    """
    if not name:
        name = target.split(":")[-1] + "_proto_java_src_extractor"

    src_jar = target.replace("_java_proto_lite", "_proto-lite-src.jar").replace(":", "/").replace("//", "")

    native.genrule(
        name = name,
        srcs = [target],
        outs = [src_out],
        cmd = """
        for FILE in $(SRCS); do
          if [[ "$$FILE" == *{0} ]]; then
            unzip -p "$$FILE" {1} > $@
            break
          fi
        done
        """.format(src_jar, src_out),
    )
    return src_out

def mediapipe_java_proto_srcs(name = ""):
    """Extracts the generated MediaPipe framework java proto source code.

    Args:
      name: The optional bazel target name.

    Returns:
      The list of the extrated MediaPipe java proto source code.
    """

    proto_src_list = []

    proto_src_list.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/framework:calculator_java_proto_lite",
        src_out = "com/google/mediapipe/proto/CalculatorProto.java",
    ))

    proto_src_list.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/framework:calculator_options_java_proto_lite",
        src_out = "com/google/mediapipe/proto/CalculatorOptionsProto.java",
    ))

    proto_src_list.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/framework:stream_handler_java_proto_lite",
        src_out = "com/google/mediapipe/proto/StreamHandlerProto.java",
    ))

    proto_src_list.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/framework:packet_factory_java_proto_lite",
        src_out = "com/google/mediapipe/proto/PacketFactoryProto.java",
    ))

    proto_src_list.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/framework:packet_generator_java_proto_lite",
        src_out = "com/google/mediapipe/proto/PacketGeneratorProto.java",
    ))

    proto_src_list.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/framework:status_handler_java_proto_lite",
        src_out = "com/google/mediapipe/proto/StatusHandlerProto.java",
    ))

    proto_src_list.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/framework:mediapipe_options_java_proto_lite",
        src_out = "com/google/mediapipe/proto/MediaPipeOptionsProto.java",
    ))

    proto_src_list.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/framework/formats/annotation:rasterization_java_proto_lite",
        src_out = "com/google/mediapipe/formats/annotation/proto/RasterizationProto.java",
    ))

    proto_src_list.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/framework/formats:classification_java_proto_lite",
        src_out = "com/google/mediapipe/formats/proto/ClassificationProto.java",
    ))

    proto_src_list.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/framework/formats:detection_java_proto_lite",
        src_out = "com/google/mediapipe/formats/proto/DetectionProto.java",
    ))

    proto_src_list.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/framework/formats:landmark_java_proto_lite",
        src_out = "com/google/mediapipe/formats/proto/LandmarkProto.java",
    ))

    proto_src_list.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/framework/formats:location_data_java_proto_lite",
        src_out = "com/google/mediapipe/formats/proto/LocationDataProto.java",
    ))

    proto_src_list.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/framework/formats:matrix_data_java_proto_lite",
        src_out = "com/google/mediapipe/formats/proto/MatrixDataProto.java",
    ))

    proto_src_list.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/framework/formats:rect_java_proto_lite",
        src_out = "com/google/mediapipe/formats/proto/RectProto.java",
    ))

    proto_src_list.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/util:color_java_proto_lite",
        src_out = "com/google/mediapipe/util/proto/ColorProto.java",
    ))

    proto_src_list.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/util:label_map_java_proto_lite",
        src_out = "com/google/mediapipe/util/proto/LabelMapProto.java",
    ))

    proto_src_list.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/util:render_data_java_proto_lite",
        src_out = "com/google/mediapipe/util/proto/RenderDataProto.java",
    ))

    return proto_src_list

def mediapipe_logging_java_proto_srcs(name = ""):
    """Extracts the generated logging-related MediaPipe java proto source code.

    Args:
      name: The optional bazel target name.

    Returns:
      The list of the extrated MediaPipe logging-related java proto source code.
    """

    proto_src_list = []

    proto_src_list.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/util/analytics:mediapipe_log_extension_java_proto_lite",
        src_out = "com/google/mediapipe/proto/MediaPipeLoggingProto.java",
    ))

    proto_src_list.append(mediapipe_java_proto_src_extractor(
        target = "//mediapipe/util/analytics:mediapipe_logging_enums_java_proto_lite",
        src_out = "com/google/mediapipe/proto/MediaPipeLoggingEnumsProto.java",
    ))
    return proto_src_list

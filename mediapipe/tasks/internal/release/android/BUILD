load("//mediapipe:version.bzl", "MEDIAPIPE_FULL_VERSION")
load("//mediapipe/tasks/internal/release/android:maven_repo.bzl", "maven_artifact", "maven_repository")

sh_binary(
    name = "maven_artifact",
    srcs = ["maven_artifact.sh"],
)

maven_artifact(
    name = "tasks_core_maven_artifact",
    src = "//mediapipe/tasks/java/com/google/mediapipe/tasks/core:tasks_core.aar",
    artifact_deps = [
        "com.google.flogger:flogger:0.6",
        "com.google.flogger:flogger-system-backend:0.6",
        "com.google.guava:guava:27.0.1-android",
        "com.google.protobuf:protobuf-javalite:4.26.1",
        "com.google.android.datatransport:transport-api:3.0.0",
        "com.google.android.datatransport:transport-backend-cct:3.1.0",
        "com.google.android.datatransport:transport-runtime:3.1.0",
    ],
    artifact_id = "tasks-core",
    group_id = "com.google.mediapipe",
    inception_year = "2022",
    lib_description = "The MediaPipe Tasks core components.",
    lib_name = "MediaPipe Taksks Core",
    lib_url = "https://mediapipe.dev/",
    version = MEDIAPIPE_FULL_VERSION,
)

maven_artifact(
    name = "tasks_audio_maven_artifact",
    src = "//mediapipe/tasks/java/com/google/mediapipe/tasks/audio:tasks_audio.aar",
    artifact_deps = [
        "androidx.annotation:annotation:1.1.0",
        "com.google.auto.value:auto-value-annotations:1.8.1",
        "com.google.auto.value:auto-value:1.8.1",
        "com.google.guava:guava:27.0.1-android",
        "com.google.mediapipe:tasks-core:" + MEDIAPIPE_FULL_VERSION,
    ],
    artifact_id = "tasks-audio",
    group_id = "com.google.mediapipe",
    inception_year = "2022",
    lib_description = "The MediaPipe Tasks Audio APIs.",
    lib_name = "MediaPipe Tasks Audio",
    lib_url = "https://mediapipe.dev/",
    version = MEDIAPIPE_FULL_VERSION,
)

maven_artifact(
    name = "tasks_vision_maven_artifact",
    src = "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision:tasks_vision.aar",
    artifact_deps = [
        "androidx.annotation:annotation:1.1.0",
        "com.google.auto.value:auto-value-annotations:1.8.1",
        "com.google.auto.value:auto-value:1.8.1",
        "com.google.guava:guava:27.0.1-android",
        "com.google.mediapipe:tasks-core:" + MEDIAPIPE_FULL_VERSION,
    ],
    artifact_id = "tasks-vision",
    group_id = "com.google.mediapipe",
    inception_year = "2022",
    lib_description = "The MediaPipe Tasks Vision APIs.",
    lib_name = "MediaPipe Tasks Vision",
    lib_url = "https://mediapipe.dev/",
    version = MEDIAPIPE_FULL_VERSION,
)

maven_artifact(
    name = "tasks_text_maven_artifact",
    src = "//mediapipe/tasks/java/com/google/mediapipe/tasks/text:tasks_text.aar",
    artifact_deps = [
        "androidx.annotation:annotation:1.1.0",
        "com.google.auto.value:auto-value-annotations:1.8.1",
        "com.google.auto.value:auto-value:1.8.1",
        "com.google.guava:guava:27.0.1-android",
        "com.google.mediapipe:tasks-core:" + MEDIAPIPE_FULL_VERSION,
    ],
    artifact_id = "tasks-text",
    group_id = "com.google.mediapipe",
    inception_year = "2022",
    lib_description = "The MediaPipe Tasks Text APIs.",
    lib_name = "MediaPipe Tasks Text",
    lib_url = "https://mediapipe.dev/",
    version = MEDIAPIPE_FULL_VERSION,
)

maven_artifact(
    name = "tasks_vision_image_generator_maven_artifact",
    src = "//mediapipe/tasks/java/com/google/mediapipe/tasks/vision/imagegenerator:tasks_vision_image_generator.aar",
    artifact_deps = [
        "androidx.annotation:annotation:1.1.0",
        "com.google.auto.value:auto-value-annotations:1.8.1",
        "com.google.auto.value:auto-value:1.8.1",
        "com.google.guava:guava:27.0.1-android",
        "com.google.mediapipe:tasks-core:" + MEDIAPIPE_FULL_VERSION,
    ],
    artifact_id = "tasks-vision-image-generator",
    group_id = "com.google.mediapipe",
    inception_year = "2023",
    lib_description = "The MediaPipe Tasks Vision Image Generator Android API.",
    lib_name = "MediaPipe Tasks Vision Image Generator",
    lib_url = "https://mediapipe.dev/",
    version = MEDIAPIPE_FULL_VERSION,
)

maven_artifact(
    name = "tasks_genai_maven_artifact",
    src = "//mediapipe/tasks/java/com/google/mediapipe/tasks/genai:tasks_genai.aar",
    artifact_deps = [
        "androidx.annotation:annotation:1.1.0",
        "com.google.auto.value:auto-value-annotations:1.8.1",
        "com.google.auto.value:auto-value:1.8.1",
        "com.google.guava:guava:27.0.1-android",
        "com.google.mediapipe:tasks-core:" + MEDIAPIPE_FULL_VERSION,
    ],
    artifact_id = "tasks-genai",
    group_id = "com.google.mediapipe",
    inception_year = "2023",
    lib_description = "The MediaPipe Tasks GenAI Android API.",
    lib_name = "MediaPipe Tasks Text GenAI",
    lib_url = "https://mediapipe.dev/",
    version = MEDIAPIPE_FULL_VERSION,
)

maven_repository(
    name = "image_generator_m2repository",
    srcs = [
        ":tasks_vision_image_generator_maven_artifact",
    ],
)

maven_repository(
    name = "genai_m2repository",
    srcs = [
        ":tasks_genai_maven_artifact",
    ],
)

maven_repository(
    name = "m2repository",
    srcs = [
        ":tasks_audio_maven_artifact",
        ":tasks_core_maven_artifact",
        ":tasks_text_maven_artifact",
        ":tasks_vision_maven_artifact",
    ],
)

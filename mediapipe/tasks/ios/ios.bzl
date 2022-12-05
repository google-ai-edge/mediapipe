"""Mediapipe Task Library Helper Rules for iOS"""

MPP_TASK_MINIMUM_OS_VERSION = "11.0"

# Default tags for filtering iOS targets. Targets are restricted to Apple platforms.
MPP_TASK_DEFAULT_TAGS = [
    "apple",
]

# Following sanitizer tests are not supported by iOS test targets.
MPP_TASK_DISABLED_SANITIZER_TAGS = [
    "noasan",
    "nomsan",
    "notsan",
]

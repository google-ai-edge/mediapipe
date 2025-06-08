# Build Custom MediaPipe Pose Landmark Tasks for Ditto

This document provides step-by-step instructions to create custom MediaPipe pose landmark task bundles **without temporal smoothing** for reduced latency processing.

## Overview

We create custom `.task` files that bypass MediaPipe's temporal filtering to achieve:
- Reduced processing latency
- Raw landmark positions without smoothing
- Compatibility with MediaPipe Tasks API
- Three complexity variants: Lite, Full, Heavy

## Prerequisites

- Bazel/Bazelisk installed
- MediaPipe repository cloned
- macOS with Xcode 16+ (for zlib compatibility fix)

## Step 1: Fix zlib Compatibility (macOS Xcode 16+)

MediaPipe's bundled zlib 1.2.13 has macro conflicts with newer macOS SDKs. Upgrade to 1.3.1:

```bash
# Edit WORKSPACE file - replace existing zlib definition with:
http_archive(
    name = "zlib",
    build_file = "@//third_party:zlib.BUILD",
    patch_args = ["-p1"],
    patches = ["@//third_party:zlib.diff"],
    sha256 = "9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23",
    strip_prefix = "zlib-1.3.1",
    url = "https://zlib.net/zlib-1.3.1.tar.gz",
)

# Clean and verify
bazelisk clean --expunge
bazelisk build -c opt //mediapipe/modules/pose_landmark:pose_landmark_filtering
```

## Step 2: Create Custom Module Directory

```bash
mkdir -p mediapipe/modules/custom_pose_landmark/pose_landmark
```

## Step 3: Copy Base Files

Copy the original pose landmark module files to our custom location:

```bash
cp mediapipe/modules/pose_landmark/* mediapipe/modules/custom_pose_landmark/pose_landmark/
```

## Step 4: Create No-Filter Graph

Create `mediapipe/modules/custom_pose_landmark/pose_landmark/pose_landmark_filtering_nofilter.pbtxt`:

Key changes from the original:
1. **Remove ENABLE dependency**: Delete `input_side_packet: "ENABLE:enable"`
2. **Replace SwitchContainer nodes** with direct no_filter calculators
3. **Remove temporal smoothing** from all 4 filtering stages:
   - Pose landmark visibilities
   - Pose landmark coordinates  
   - World landmark visibilities
   - World landmark coordinates
4. **Disable auxiliary landmark filtering**

The result is a graph that passes landmarks through without any temporal smoothing.

## Step 5: Update BUILD File

Add the no-filter subgraph to `mediapipe/modules/custom_pose_landmark/pose_landmark/BUILD`:

```python
mediapipe_simple_subgraph(
    name = "pose_landmark_filtering_nofilter_subgraph",
    graph = "pose_landmark_filtering_nofilter.pbtxt",
    register_as = "PoseLandmarkFilteringNoFilter",
    deps = [
        "//mediapipe/calculators/util:alignment_points_to_rects_calculator",
        "//mediapipe/calculators/util:landmarks_smoothing_calculator",
        "//mediapipe/calculators/util:landmarks_to_detection_calculator",
        "//mediapipe/calculators/util:visibility_smoothing_calculator",
    ],
)
```

## Step 6: Build and Verify Custom Graph

```bash
bazelisk build -c opt //mediapipe/modules/custom_pose_landmark/pose_landmark:pose_landmark_filtering_nofilter_subgraph

# Verify no SwitchContainer references remain
! grep -q "SwitchContainer" mediapipe/modules/custom_pose_landmark/pose_landmark/pose_landmark_filtering_nofilter.pbtxt

# Verify output streams are preserved
grep "^output_stream:" mediapipe/modules/custom_pose_landmark/pose_landmark/pose_landmark_filtering_nofilter.pbtxt
```

Expected output streams:
- `FILTERED_NORM_LANDMARKS`
- `FILTERED_AUX_NORM_LANDMARKS` 
- `FILTERED_WORLD_LANDMARKS`

## Step 7: Download Official MediaPipe Models

```bash
# Create models directory
mkdir -p models/pose && cd models/pose

# Download official task bundles
wget -O pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

wget -O pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task

wget -O pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

# Extract TFLite models from task bundles
mkdir -p lite full heavy
unzip -j pose_landmarker_lite.task -d lite
unzip -j pose_landmarker_full.task -d full  
unzip -j pose_landmarker_heavy.task -d heavy
```

## Step 8: Organize Models in Custom Module

```bash
cd mediapipe/modules/custom_pose_landmark/pose_landmark
mkdir -p models/{lite,full,heavy}

# Copy extracted models
cp ../../../../models/pose/lite/*.tflite models/lite/
cp ../../../../models/pose/full/*.tflite models/full/
cp ../../../../models/pose/heavy/*.tflite models/heavy/
```

## Step 9: Create Task Configuration Files

Create base options for each model variant:

### Lite Version
`custom_pose_landmark_lite_nofilter_base_options.textproto`:
```protobuf
file_name: "custom_pose_landmarker_lite_nofilter.task"

model_asset {
  file_name: "models/lite/pose_detector.tflite"
}

model_asset {
  file_name: "models/lite/pose_landmarks_detector.tflite"
}

graph_options {
  use_gpu: false
  enable_segmentation: false
  model_complexity: 0  # Lite version
}
```

### Full Version  
`custom_pose_landmark_full_nofilter_base_options.textproto`:
```protobuf
file_name: "custom_pose_landmarker_full_nofilter.task"

model_asset {
  file_name: "models/full/pose_detector.tflite"
}

model_asset {
  file_name: "models/full/pose_landmarks_detector.tflite"
}

graph_options {
  use_gpu: false
  enable_segmentation: false
  model_complexity: 1  # Full version
}
```

### Heavy Version
`custom_pose_landmark_heavy_nofilter_base_options.textproto`:
```protobuf
file_name: "custom_pose_landmarker_heavy_nofilter.task"

model_asset {
  file_name: "models/heavy/pose_detector.tflite"
}

model_asset {
  file_name: "models/heavy/pose_landmarks_detector.tflite"
}

graph_options {
  use_gpu: false
  enable_segmentation: false
  model_complexity: 2  # Heavy version
}
```

## Step 10: Build Task Bundles

From the repository root:

```bash
# Build the custom graph first
bazelisk build -c opt //mediapipe/modules/custom_pose_landmark/pose_landmark:pose_landmark_filtering_nofilter_subgraph

# Create task bundles (ZIP files with specific structure)
cd mediapipe/modules/custom_pose_landmark/pose_landmark

# Lite bundle (~5.5MB)
zip -j custom_pose_landmarker_lite_nofilter.task \
  models/lite/*.tflite \
  custom_pose_landmark_lite_nofilter_base_options.textproto \
  ../../../../bazel-bin/mediapipe/modules/custom_pose_landmark/pose_landmark/pose_landmark_filtering_nofilter_subgraph.binarypb

# Full bundle (~9.0MB)  
zip -j custom_pose_landmarker_full_nofilter.task \
  models/full/*.tflite \
  custom_pose_landmark_full_nofilter_base_options.textproto \
  ../../../../bazel-bin/mediapipe/modules/custom_pose_landmark/pose_landmark/pose_landmark_filtering_nofilter_subgraph.binarypb

# Heavy bundle (~29MB)
zip -j custom_pose_landmarker_heavy_nofilter.task \
  models/heavy/*.tflite \
  custom_pose_landmark_heavy_nofilter_base_options.textproto \
  ../../../../bazel-bin/mediapipe/modules/custom_pose_landmark/pose_landmark/pose_landmark_filtering_nofilter_subgraph.binarypb
```

## Step 11: Verify Task Bundles

```bash
# Check bundle contents
unzip -l custom_pose_landmarker_lite_nofilter.task
unzip -l custom_pose_landmarker_full_nofilter.task  
unzip -l custom_pose_landmarker_heavy_nofilter.task
```

Each bundle should contain:
- `pose_detector.tflite` (2.9MB)
- `pose_landmarks_detector.tflite` (varies by complexity)
- Base options textproto file
- Binary graph file (`pose_landmark_filtering_nofilter_subgraph.binarypb`)

## Usage in Applications

### Android/Kotlin
```kotlin
val base = BaseOptions.builder()
    .setModelAssetPath("custom_pose_landmarker_full_nofilter.task")
    .build()

val landmarker = PoseLandmarker.createFromOptions(
    FilesetResolver.forVisionTasks(context),
    PoseLandmarker.PoseLandmarkerOptions.builder()
        .setBaseOptions(base)
        .setRunningMode(RunningMode.LIVE_STREAM)
        .build())
```

### iOS/Swift
```swift
let baseOptions = BaseOptions()
baseOptions.modelAssetPath = "custom_pose_landmarker_full_nofilter.task"

let options = PoseLandmarkerOptions()
options.baseOptions = baseOptions
options.runningMode = .liveStream

let landmarker = try PoseLandmarker(options: options)
```

## Model Comparison

| Variant | Size | Speed | Accuracy | Use Case |
|---------|------|-------|----------|-----------|
| Lite    | 5.5MB | Fastest | Good | Mobile devices, real-time |
| Full    | 9.0MB | Balanced | Better | General applications |
| Heavy   | 29MB | Slower | Best | High-accuracy requirements |

## Benefits of No-Filter Version

- **Reduced Latency**: No temporal smoothing overhead
- **Raw Positions**: Unfiltered landmark coordinates  
- **Real-time Responsiveness**: Immediate response to movement
- **Same API**: Drop-in replacement for standard MediaPipe tasks

## Files Created

```
mediapipe/modules/custom_pose_landmark/pose_landmark/
├── pose_landmark_filtering_nofilter.pbtxt
├── custom_pose_landmark_lite_nofilter_base_options.textproto
├── custom_pose_landmark_full_nofilter_base_options.textproto
├── custom_pose_landmark_heavy_nofilter_base_options.textproto
├── custom_pose_landmarker_lite_nofilter.task
├── custom_pose_landmarker_full_nofilter.task
├── custom_pose_landmarker_heavy_nofilter.task
└── models/
    ├── lite/
    ├── full/
    └── heavy/
```

## Troubleshooting

### Build Errors
- Ensure zlib 1.3.1 upgrade is applied for macOS Xcode 16+
- Run `bazelisk clean --expunge` if encountering cache issues
- Verify all file paths are correct

### Task Bundle Issues  
- Task bundles are ZIP files - verify they can be opened
- Check that all required files are included in bundle
- Ensure binary graph was built successfully before bundling

### API Usage
- Task files must be accessible to your application
- Use the exact `.task` filename as specified in base options
- Verify MediaPipe Tasks API version compatibility

## Repository Structure

This creates a self-contained custom module that doesn't modify the original MediaPipe pose landmark files, making it easy to maintain and update. 

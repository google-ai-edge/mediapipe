# Custom Pose Landmark Module

This is a custom fork of the MediaPipe pose landmark module with modified filtering behavior.

## Why This Fork Exists

This custom module was created to provide pose landmark detection **without temporal smoothing**. The original MediaPipe pose landmark filtering applies temporal smoothing (low-pass filters and One-Euro filters) to reduce jitter, but this introduces latency and can be undesirable for applications requiring the most responsive, unfiltered landmark positions.

## Subgraphs

Subgraphs|Details
:--- | :---
[`PoseLandmarkByRoiCpu`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/pose_landmark/pose_landmark_by_roi_cpu.pbtxt)| Detects landmarks of a single body pose. See landmarks (aka keypoints) [scheme](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/pose_landmark/pose_landmark_topology.svg). (CPU input, and inference is executed on CPU.)
[`PoseLandmarkByRoiGpu`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/pose_landmark/pose_landmark_by_roi_gpu.pbtxt)| Detects landmarks of a single body pose. See landmarks (aka keypoints) [scheme](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/pose_landmark/pose_landmark_topology.svg). (GPU input, and inference is executed on GPU)
[`PoseLandmarkCpu`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/pose_landmark/pose_landmark_cpu.pbtxt)| Detects landmarks of a single body pose. See landmarks (aka keypoints) [scheme](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/pose_landmark/pose_landmark_topology.svg). (CPU input, and inference is executed on CPU)
[`PoseLandmarkGpu`](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/pose_landmark/pose_landmark_gpu.pbtxt)| Detects landmarks of a single body pose. See landmarks (aka keypoints) [scheme](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/modules/pose_landmark/pose_landmark_topology.svg). (GPU input, and inference is executed on GPU.)
[`PoseLandmarkFilteringNoFilter`](pose_landmark_filtering_nofilter.pbtxt)| **Custom**: Landmark filtering without temporal smoothing. Provides raw, unfiltered landmark positions for applications requiring minimal latency.

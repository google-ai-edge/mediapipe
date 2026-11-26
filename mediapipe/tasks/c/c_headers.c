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

// Includes the MediaPipe Tasks C API headers, to verify that they are valid
// C99 when they are used together. This file is compiled as C99 with
// `-pedantic-errors`; it contains no code of its own.
//
// Add every new C API header to this list.

#ifdef __cplusplus
#error "This file must be compiled as C, not as C++."
#endif

#include "mediapipe/tasks/c/audio/audio_classifier/audio_classifier.h"
#include "mediapipe/tasks/c/audio/audio_embedder/audio_embedder.h"
#include "mediapipe/tasks/c/audio/audio_transcriber/audio_transcriber.h"
#include "mediapipe/tasks/c/audio/core/common.h"
#include "mediapipe/tasks/c/audio/core/transcription_segment.h"
#include "mediapipe/tasks/c/audio/text_to_speech/text_to_speech.h"
#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/components/containers/classification_result.h"
#include "mediapipe/tasks/c/components/containers/detection_result.h"
#include "mediapipe/tasks/c/components/containers/embedding_result.h"
#include "mediapipe/tasks/c/components/containers/keypoint.h"
#include "mediapipe/tasks/c/components/containers/landmark.h"
#include "mediapipe/tasks/c/components/containers/matrix.h"
#include "mediapipe/tasks/c/components/containers/rect.h"
#include "mediapipe/tasks/c/components/processors/classifier_options.h"
#include "mediapipe/tasks/c/components/processors/embedder_options.h"
#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/common.h"
#include "mediapipe/tasks/c/core/mp_logging.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/decision/decision_maker/decision_maker.h"
#include "mediapipe/tasks/c/metadata/flatbuffer_api.h"
#include "mediapipe/tasks/c/retrieval/semantic_retriever/semantic_retriever.h"
#include "mediapipe/tasks/c/retrieval/universal_embedder/universal_embedder.h"
#include "mediapipe/tasks/c/text/language_detector/language_detector.h"
#include "mediapipe/tasks/c/text/text_classifier/text_classifier.h"
#include "mediapipe/tasks/c/text/text_embedder/text_embedder.h"
#include "mediapipe/tasks/c/text/text_proofreader/text_proofreader.h"
#include "mediapipe/tasks/c/text/text_summarizer/text_summarizer.h"
#include "mediapipe/tasks/c/vision/core/image.h"
#include "mediapipe/tasks/c/vision/core/image_processing_options.h"
#include "mediapipe/tasks/c/vision/face_detector/face_detector.h"
#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker.h"
#include "mediapipe/tasks/c/vision/face_landmarker/face_landmarker_result.h"
#include "mediapipe/tasks/c/vision/gesture_recognizer/gesture_recognizer.h"
#include "mediapipe/tasks/c/vision/gesture_recognizer/gesture_recognizer_result.h"
#include "mediapipe/tasks/c/vision/hand_landmarker/hand_landmarker.h"
#include "mediapipe/tasks/c/vision/hand_landmarker/hand_landmarker_result.h"
#include "mediapipe/tasks/c/vision/holistic_landmarker/holistic_landmarker.h"
#include "mediapipe/tasks/c/vision/holistic_landmarker/holistic_landmarker_result.h"
#include "mediapipe/tasks/c/vision/image_classifier/image_classifier.h"
#include "mediapipe/tasks/c/vision/image_embedder/image_embedder.h"
#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter.h"
#include "mediapipe/tasks/c/vision/image_segmenter/image_segmenter_result.h"
#include "mediapipe/tasks/c/vision/interactive_segmenter/interactive_segmenter.h"
#include "mediapipe/tasks/c/vision/interactive_segmenter_legacy/interactive_segmenter_legacy.h"
#include "mediapipe/tasks/c/vision/object_detector/object_detector.h"
#include "mediapipe/tasks/c/vision/pose_landmarker/pose_landmarker.h"
#include "mediapipe/tasks/c/vision/pose_landmarker/pose_landmarker_result.h"

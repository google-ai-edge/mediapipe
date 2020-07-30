// Copyright 2019 The MediaPipe Authors.
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

#include <unordered_map>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/formats/object_detection/anchor.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/tflite/config.h"
#include "tensorflow/lite/interpreter.h"

#if MEDIAPIPE_TFLITE_GL_INFERENCE
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_buffer.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_shader.h"
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE

#if MEDIAPIPE_TFLITE_METAL_INFERENCE
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#import "mediapipe/gpu/MPPMetalHelper.h"
#include "mediapipe/gpu/MPPMetalUtil.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"
#endif  // MEDIAPIPE_TFLITE_METAL_INFERENCE

namespace {
constexpr int kNumInputTensorsWithAnchors = 3;
constexpr int kNumCoordsPerBox = 4;

constexpr char kTensorsTag[] = "TENSORS";
constexpr char kTensorsGpuTag[] = "TENSORS_GPU";
}  // namespace

namespace mediapipe {

#if MEDIAPIPE_TFLITE_GL_INFERENCE
using ::tflite::gpu::gl::CreateReadWriteShaderStorageBuffer;
using ::tflite::gpu::gl::GlShader;
typedef ::tflite::gpu::gl::GlProgram GpuProgram;
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE
typedef id<MTLComputePipelineState> GpuProgram;
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE

namespace {

#if MEDIAPIPE_TFLITE_GPU_SUPPORTED
struct GPUData {
  GpuProgram decode_program;
  GpuProgram score_program;
  GpuTensor decoded_boxes_buffer;
  GpuTensor raw_boxes_buffer;
  GpuTensor raw_anchors_buffer;
  GpuTensor scored_boxes_buffer;
  GpuTensor raw_scores_buffer;
};
#endif  // MEDIAPIPE_TFLITE_GPU_SUPPORTED

void ConvertRawValuesToAnchors(const float* raw_anchors, int num_boxes,
                               std::vector<Anchor>* anchors) {
  anchors->clear();
  for (int i = 0; i < num_boxes; ++i) {
    Anchor new_anchor;
    new_anchor.set_y_center(raw_anchors[i * kNumCoordsPerBox + 0]);
    new_anchor.set_x_center(raw_anchors[i * kNumCoordsPerBox + 1]);
    new_anchor.set_h(raw_anchors[i * kNumCoordsPerBox + 2]);
    new_anchor.set_w(raw_anchors[i * kNumCoordsPerBox + 3]);
    anchors->push_back(new_anchor);
  }
}

void ConvertAnchorsToRawValues(const std::vector<Anchor>& anchors,
                               int num_boxes, float* raw_anchors) {
  CHECK_EQ(anchors.size(), num_boxes);
  int box = 0;
  for (const auto& anchor : anchors) {
    raw_anchors[box * kNumCoordsPerBox + 0] = anchor.y_center();
    raw_anchors[box * kNumCoordsPerBox + 1] = anchor.x_center();
    raw_anchors[box * kNumCoordsPerBox + 2] = anchor.h();
    raw_anchors[box * kNumCoordsPerBox + 3] = anchor.w();
    ++box;
  }
}

}  // namespace

// Convert result TFLite tensors from object detection models into MediaPipe
// Detections.
//
// Input:
//  TENSORS - Vector of TfLiteTensor of type kTfLiteFloat32. The vector of
//               tensors can have 2 or 3 tensors. First tensor is the predicted
//               raw boxes/keypoints. The size of the values must be (num_boxes
//               * num_predicted_values). Second tensor is the score tensor. The
//               size of the valuse must be (num_boxes * num_classes). It's
//               optional to pass in a third tensor for anchors (e.g. for SSD
//               models) depend on the outputs of the detection model. The size
//               of anchor tensor must be (num_boxes * 4).
//  TENSORS_GPU - vector of GlBuffer of MTLBuffer.
// Output:
//  DETECTIONS - Result MediaPipe detections.
//
// Usage example:
// node {
//   calculator: "TfLiteTensorsToDetectionsCalculator"
//   input_stream: "TENSORS:tensors"
//   input_side_packet: "ANCHORS:anchors"
//   output_stream: "DETECTIONS:detections"
//   options: {
//     [mediapipe.TfLiteTensorsToDetectionsCalculatorOptions.ext] {
//       num_classes: 91
//       num_boxes: 1917
//       num_coords: 4
//       ignore_classes: [0, 1, 2]
//       x_scale: 10.0
//       y_scale: 10.0
//       h_scale: 5.0
//       w_scale: 5.0
//     }
//   }
// }
class TfLiteTensorsToDetectionsCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  ::mediapipe::Status Close(CalculatorContext* cc) override;

 private:
  ::mediapipe::Status ProcessCPU(CalculatorContext* cc,
                                 std::vector<Detection>* output_detections);
  ::mediapipe::Status ProcessGPU(CalculatorContext* cc,
                                 std::vector<Detection>* output_detections);

  ::mediapipe::Status LoadOptions(CalculatorContext* cc);
  ::mediapipe::Status GpuInit(CalculatorContext* cc);
  ::mediapipe::Status DecodeBoxes(const float* raw_boxes,
                                  const std::vector<Anchor>& anchors,
                                  std::vector<float>* boxes);
  ::mediapipe::Status ConvertToDetections(
      const float* detection_boxes, const float* detection_scores,
      const int* detection_classes, std::vector<Detection>* output_detections);
  Detection ConvertToDetection(float box_ymin, float box_xmin, float box_ymax,
                               float box_xmax, float score, int class_id,
                               bool flip_vertically);

  int num_classes_ = 0;
  int num_boxes_ = 0;
  int num_coords_ = 0;
  std::set<int> ignore_classes_;

  ::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions options_;
  std::vector<Anchor> anchors_;
  bool side_packet_anchors_{};

#if MEDIAPIPE_TFLITE_GL_INFERENCE
  mediapipe::GlCalculatorHelper gpu_helper_;
  std::unique_ptr<GPUData> gpu_data_;
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE
  MPPMetalHelper* gpu_helper_ = nullptr;
  std::unique_ptr<GPUData> gpu_data_;
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE

  bool gpu_input_ = false;
  bool anchors_init_ = false;
};
REGISTER_CALCULATOR(TfLiteTensorsToDetectionsCalculator);

::mediapipe::Status TfLiteTensorsToDetectionsCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(!cc->Inputs().GetTags().empty());
  RET_CHECK(!cc->Outputs().GetTags().empty());

  bool use_gpu = false;

  if (cc->Inputs().HasTag(kTensorsTag)) {
    cc->Inputs().Tag(kTensorsTag).Set<std::vector<TfLiteTensor>>();
  }

  if (cc->Inputs().HasTag(kTensorsGpuTag)) {
    cc->Inputs().Tag(kTensorsGpuTag).Set<std::vector<GpuTensor>>();
    use_gpu |= true;
  }

  if (cc->Outputs().HasTag("DETECTIONS")) {
    cc->Outputs().Tag("DETECTIONS").Set<std::vector<Detection>>();
  }

  if (cc->InputSidePackets().UsesTags()) {
    if (cc->InputSidePackets().HasTag("ANCHORS")) {
      cc->InputSidePackets().Tag("ANCHORS").Set<std::vector<Anchor>>();
    }
  }

  if (use_gpu) {
#if MEDIAPIPE_TFLITE_GL_INFERENCE
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE
    MP_RETURN_IF_ERROR([MPPMetalHelper updateContract:cc]);
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteTensorsToDetectionsCalculator::Open(
    CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  if (cc->Inputs().HasTag(kTensorsGpuTag)) {
    gpu_input_ = true;
#if MEDIAPIPE_TFLITE_GL_INFERENCE
    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE
    gpu_helper_ = [[MPPMetalHelper alloc] initWithCalculatorContext:cc];
    RET_CHECK(gpu_helper_);
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE
  }

  MP_RETURN_IF_ERROR(LoadOptions(cc));
  side_packet_anchors_ = cc->InputSidePackets().HasTag("ANCHORS");

  if (gpu_input_) {
    MP_RETURN_IF_ERROR(GpuInit(cc));
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteTensorsToDetectionsCalculator::Process(
    CalculatorContext* cc) {
  if ((!gpu_input_ && cc->Inputs().Tag(kTensorsTag).IsEmpty()) ||
      (gpu_input_ && cc->Inputs().Tag(kTensorsGpuTag).IsEmpty())) {
    return ::mediapipe::OkStatus();
  }

  auto output_detections = absl::make_unique<std::vector<Detection>>();

  if (gpu_input_) {
    MP_RETURN_IF_ERROR(ProcessGPU(cc, output_detections.get()));
  } else {
    MP_RETURN_IF_ERROR(ProcessCPU(cc, output_detections.get()));
  }

  // Output
  if (cc->Outputs().HasTag("DETECTIONS")) {
    cc->Outputs()
        .Tag("DETECTIONS")
        .Add(output_detections.release(), cc->InputTimestamp());
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteTensorsToDetectionsCalculator::ProcessCPU(
    CalculatorContext* cc, std::vector<Detection>* output_detections) {
  const auto& input_tensors =
      cc->Inputs().Tag(kTensorsTag).Get<std::vector<TfLiteTensor>>();

  if (input_tensors.size() == 2 ||
      input_tensors.size() == kNumInputTensorsWithAnchors) {
    // Postprocessing on CPU for model without postprocessing op. E.g. output
    // raw score tensor and box tensor. Anchor decoding will be handled below.
    const TfLiteTensor* raw_box_tensor = &input_tensors[0];
    const TfLiteTensor* raw_score_tensor = &input_tensors[1];

    // TODO: Add flexible input tensor size handling.
    CHECK_EQ(raw_box_tensor->dims->size, 3);
    CHECK_EQ(raw_box_tensor->dims->data[0], 1);
    CHECK_EQ(raw_box_tensor->dims->data[1], num_boxes_);
    CHECK_EQ(raw_box_tensor->dims->data[2], num_coords_);
    CHECK_EQ(raw_score_tensor->dims->size, 3);
    CHECK_EQ(raw_score_tensor->dims->data[0], 1);
    CHECK_EQ(raw_score_tensor->dims->data[1], num_boxes_);
    CHECK_EQ(raw_score_tensor->dims->data[2], num_classes_);
    const float* raw_boxes = raw_box_tensor->data.f;
    const float* raw_scores = raw_score_tensor->data.f;

    // TODO: Support other options to load anchors.
    if (!anchors_init_) {
      if (input_tensors.size() == kNumInputTensorsWithAnchors) {
        const TfLiteTensor* anchor_tensor = &input_tensors[2];
        CHECK_EQ(anchor_tensor->dims->size, 2);
        CHECK_EQ(anchor_tensor->dims->data[0], num_boxes_);
        CHECK_EQ(anchor_tensor->dims->data[1], kNumCoordsPerBox);
        const float* raw_anchors = anchor_tensor->data.f;
        ConvertRawValuesToAnchors(raw_anchors, num_boxes_, &anchors_);
      } else if (side_packet_anchors_) {
        CHECK(!cc->InputSidePackets().Tag("ANCHORS").IsEmpty());
        anchors_ =
            cc->InputSidePackets().Tag("ANCHORS").Get<std::vector<Anchor>>();
      } else {
        return ::mediapipe::UnavailableError("No anchor data available.");
      }
      anchors_init_ = true;
    }
    std::vector<float> boxes(num_boxes_ * num_coords_);
    MP_RETURN_IF_ERROR(DecodeBoxes(raw_boxes, anchors_, &boxes));

    std::vector<float> detection_scores(num_boxes_);
    std::vector<int> detection_classes(num_boxes_);

    // Filter classes by scores.
    for (int i = 0; i < num_boxes_; ++i) {
      int class_id = -1;
      float max_score = -std::numeric_limits<float>::max();
      // Find the top score for box i.
      for (int score_idx = 0; score_idx < num_classes_; ++score_idx) {
        if (ignore_classes_.find(score_idx) == ignore_classes_.end()) {
          auto score = raw_scores[i * num_classes_ + score_idx];
          if (options_.sigmoid_score()) {
            if (options_.has_score_clipping_thresh()) {
              score = score < -options_.score_clipping_thresh()
                          ? -options_.score_clipping_thresh()
                          : score;
              score = score > options_.score_clipping_thresh()
                          ? options_.score_clipping_thresh()
                          : score;
            }
            score = 1.0f / (1.0f + std::exp(-score));
          }
          if (max_score < score) {
            max_score = score;
            class_id = score_idx;
          }
        }
      }
      detection_scores[i] = max_score;
      detection_classes[i] = class_id;
    }

    MP_RETURN_IF_ERROR(
        ConvertToDetections(boxes.data(), detection_scores.data(),
                            detection_classes.data(), output_detections));
  } else {
    // Postprocessing on CPU with postprocessing op (e.g. anchor decoding and
    // non-maximum suppression) within the model.
    RET_CHECK_EQ(input_tensors.size(), 4);

    const TfLiteTensor* detection_boxes_tensor = &input_tensors[0];
    const TfLiteTensor* detection_classes_tensor = &input_tensors[1];
    const TfLiteTensor* detection_scores_tensor = &input_tensors[2];
    const TfLiteTensor* num_boxes_tensor = &input_tensors[3];
    RET_CHECK_EQ(num_boxes_tensor->dims->size, 1);
    RET_CHECK_EQ(num_boxes_tensor->dims->data[0], 1);
    const float* num_boxes = num_boxes_tensor->data.f;
    num_boxes_ = num_boxes[0];
    RET_CHECK_EQ(detection_boxes_tensor->dims->size, 3);
    RET_CHECK_EQ(detection_boxes_tensor->dims->data[0], 1);
    const int max_detections = detection_boxes_tensor->dims->data[1];
    RET_CHECK_EQ(detection_boxes_tensor->dims->data[2], num_coords_);
    RET_CHECK_EQ(detection_classes_tensor->dims->size, 2);
    RET_CHECK_EQ(detection_classes_tensor->dims->data[0], 1);
    RET_CHECK_EQ(detection_classes_tensor->dims->data[1], max_detections);
    RET_CHECK_EQ(detection_scores_tensor->dims->size, 2);
    RET_CHECK_EQ(detection_scores_tensor->dims->data[0], 1);
    RET_CHECK_EQ(detection_scores_tensor->dims->data[1], max_detections);

    const float* detection_boxes = detection_boxes_tensor->data.f;
    const float* detection_scores = detection_scores_tensor->data.f;
    std::vector<int> detection_classes(num_boxes_);
    for (int i = 0; i < num_boxes_; ++i) {
      detection_classes[i] =
          static_cast<int>(detection_classes_tensor->data.f[i]);
    }
    MP_RETURN_IF_ERROR(ConvertToDetections(detection_boxes, detection_scores,
                                           detection_classes.data(),
                                           output_detections));
  }
  return ::mediapipe::OkStatus();
}
::mediapipe::Status TfLiteTensorsToDetectionsCalculator::ProcessGPU(
    CalculatorContext* cc, std::vector<Detection>* output_detections) {
#if MEDIAPIPE_TFLITE_GL_INFERENCE
  const auto& input_tensors =
      cc->Inputs().Tag(kTensorsGpuTag).Get<std::vector<GpuTensor>>();
  RET_CHECK_GE(input_tensors.size(), 2);

  MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this, &input_tensors, &cc,
                                                 &output_detections]()
                                                    -> ::mediapipe::Status {
    // Copy inputs.
    MP_RETURN_IF_ERROR(
        CopyBuffer(input_tensors[0], gpu_data_->raw_boxes_buffer));
    MP_RETURN_IF_ERROR(
        CopyBuffer(input_tensors[1], gpu_data_->raw_scores_buffer));
    if (!anchors_init_) {
      if (side_packet_anchors_) {
        CHECK(!cc->InputSidePackets().Tag("ANCHORS").IsEmpty());
        const auto& anchors =
            cc->InputSidePackets().Tag("ANCHORS").Get<std::vector<Anchor>>();
        std::vector<float> raw_anchors(num_boxes_ * kNumCoordsPerBox);
        ConvertAnchorsToRawValues(anchors, num_boxes_, raw_anchors.data());
        MP_RETURN_IF_ERROR(gpu_data_->raw_anchors_buffer.Write<float>(
            absl::MakeSpan(raw_anchors)));
      } else {
        CHECK_EQ(input_tensors.size(), kNumInputTensorsWithAnchors);
        MP_RETURN_IF_ERROR(
            CopyBuffer(input_tensors[2], gpu_data_->raw_anchors_buffer));
      }
      anchors_init_ = true;
    }

    // Run shaders.
    // Decode boxes.
    MP_RETURN_IF_ERROR(gpu_data_->decoded_boxes_buffer.BindToIndex(0));
    MP_RETURN_IF_ERROR(gpu_data_->raw_boxes_buffer.BindToIndex(1));
    MP_RETURN_IF_ERROR(gpu_data_->raw_anchors_buffer.BindToIndex(2));
    const tflite::gpu::uint3 decode_workgroups = {num_boxes_, 1, 1};
    MP_RETURN_IF_ERROR(gpu_data_->decode_program.Dispatch(decode_workgroups));

    // Score boxes.
    MP_RETURN_IF_ERROR(gpu_data_->scored_boxes_buffer.BindToIndex(0));
    MP_RETURN_IF_ERROR(gpu_data_->raw_scores_buffer.BindToIndex(1));
    const tflite::gpu::uint3 score_workgroups = {num_boxes_, 1, 1};
    MP_RETURN_IF_ERROR(gpu_data_->score_program.Dispatch(score_workgroups));

    // Copy decoded boxes from GPU to CPU.
    std::vector<float> boxes(num_boxes_ * num_coords_);
    MP_RETURN_IF_ERROR(
        gpu_data_->decoded_boxes_buffer.Read(absl::MakeSpan(boxes)));
    std::vector<float> score_class_id_pairs(num_boxes_ * 2);
    MP_RETURN_IF_ERROR(gpu_data_->scored_boxes_buffer.Read(
        absl::MakeSpan(score_class_id_pairs)));

    // TODO: b/138851969. Is it possible to output a float vector
    // for score and an int vector for class so that we can avoid copying twice?
    std::vector<float> detection_scores(num_boxes_);
    std::vector<int> detection_classes(num_boxes_);
    for (int i = 0; i < num_boxes_; ++i) {
      detection_scores[i] = score_class_id_pairs[i * 2];
      detection_classes[i] = static_cast<int>(score_class_id_pairs[i * 2 + 1]);
    }
    MP_RETURN_IF_ERROR(
        ConvertToDetections(boxes.data(), detection_scores.data(),
                            detection_classes.data(), output_detections));

    return ::mediapipe::OkStatus();
  }));
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE

  const auto& input_tensors =
      cc->Inputs().Tag(kTensorsGpuTag).Get<std::vector<GpuTensor>>();
  RET_CHECK_GE(input_tensors.size(), 2);

  // Copy inputs.
  [MPPMetalUtil blitMetalBufferTo:gpu_data_->raw_boxes_buffer
                             from:input_tensors[0]
                         blocking:false
                    commandBuffer:[gpu_helper_ commandBuffer]];
  [MPPMetalUtil blitMetalBufferTo:gpu_data_->raw_scores_buffer
                             from:input_tensors[1]
                         blocking:false
                    commandBuffer:[gpu_helper_ commandBuffer]];
  if (!anchors_init_) {
    if (side_packet_anchors_) {
      CHECK(!cc->InputSidePackets().Tag("ANCHORS").IsEmpty());
      const auto& anchors =
          cc->InputSidePackets().Tag("ANCHORS").Get<std::vector<Anchor>>();
      std::vector<float> raw_anchors(num_boxes_ * kNumCoordsPerBox);
      ConvertAnchorsToRawValues(anchors, num_boxes_, raw_anchors.data());
      memcpy([gpu_data_->raw_anchors_buffer contents], raw_anchors.data(),
             raw_anchors.size() * sizeof(float));
    } else {
      RET_CHECK_EQ(input_tensors.size(), kNumInputTensorsWithAnchors);
      [MPPMetalUtil blitMetalBufferTo:gpu_data_->raw_anchors_buffer
                                 from:input_tensors[2]
                             blocking:false
                        commandBuffer:[gpu_helper_ commandBuffer]];
    }
    anchors_init_ = true;
  }

  // Run shaders.
  id<MTLCommandBuffer> command_buffer = [gpu_helper_ commandBuffer];
  command_buffer.label = @"TfLiteDecodeAndScoreBoxes";
  id<MTLComputeCommandEncoder> command_encoder =
      [command_buffer computeCommandEncoder];
  [command_encoder setComputePipelineState:gpu_data_->decode_program];
  [command_encoder setBuffer:gpu_data_->decoded_boxes_buffer
                      offset:0
                     atIndex:0];
  [command_encoder setBuffer:gpu_data_->raw_boxes_buffer offset:0 atIndex:1];
  [command_encoder setBuffer:gpu_data_->raw_anchors_buffer offset:0 atIndex:2];
  MTLSize decode_threads_per_group = MTLSizeMake(1, 1, 1);
  MTLSize decode_threadgroups = MTLSizeMake(num_boxes_, 1, 1);
  [command_encoder dispatchThreadgroups:decode_threadgroups
                  threadsPerThreadgroup:decode_threads_per_group];

  [command_encoder setComputePipelineState:gpu_data_->score_program];
  [command_encoder setBuffer:gpu_data_->scored_boxes_buffer offset:0 atIndex:0];
  [command_encoder setBuffer:gpu_data_->raw_scores_buffer offset:0 atIndex:1];
  MTLSize score_threads_per_group = MTLSizeMake(1, num_classes_, 1);
  MTLSize score_threadgroups = MTLSizeMake(num_boxes_, 1, 1);
  [command_encoder dispatchThreadgroups:score_threadgroups
                  threadsPerThreadgroup:score_threads_per_group];
  [command_encoder endEncoding];
  [MPPMetalUtil commitCommandBufferAndWait:command_buffer];

  // Copy decoded boxes from GPU to CPU.
  std::vector<float> boxes(num_boxes_ * num_coords_);
  memcpy(boxes.data(), [gpu_data_->decoded_boxes_buffer contents],
         num_boxes_ * num_coords_ * sizeof(float));
  std::vector<float> score_class_id_pairs(num_boxes_ * 2);
  memcpy(score_class_id_pairs.data(), [gpu_data_->scored_boxes_buffer contents],
         num_boxes_ * 2 * sizeof(float));

  // Output detections.
  // TODO Adjust shader to avoid copying shader output twice.
  std::vector<float> detection_scores(num_boxes_);
  std::vector<int> detection_classes(num_boxes_);
  for (int i = 0; i < num_boxes_; ++i) {
    detection_scores[i] = score_class_id_pairs[i * 2];
    detection_classes[i] = static_cast<int>(score_class_id_pairs[i * 2 + 1]);
  }
  MP_RETURN_IF_ERROR(ConvertToDetections(boxes.data(), detection_scores.data(),
                                         detection_classes.data(),
                                         output_detections));

#else
  LOG(ERROR) << "GPU input on non-Android not supported yet.";
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE
  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteTensorsToDetectionsCalculator::Close(
    CalculatorContext* cc) {
#if MEDIAPIPE_TFLITE_GL_INFERENCE
  gpu_helper_.RunInGlContext([this] { gpu_data_.reset(); });
#elif MEDIAPIPE_TFLITE_METAL_INFERENCE
  gpu_data_.reset();
#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteTensorsToDetectionsCalculator::LoadOptions(
    CalculatorContext* cc) {
  // Get calculator options specified in the graph.
  options_ =
      cc->Options<::mediapipe::TfLiteTensorsToDetectionsCalculatorOptions>();

  num_classes_ = options_.num_classes();
  num_boxes_ = options_.num_boxes();
  num_coords_ = options_.num_coords();

  // Currently only support 2D when num_values_per_keypoint equals to 2.
  CHECK_EQ(options_.num_values_per_keypoint(), 2);

  // Check if the output size is equal to the requested boxes and keypoints.
  CHECK_EQ(options_.num_keypoints() * options_.num_values_per_keypoint() +
               kNumCoordsPerBox,
           num_coords_);

  for (int i = 0; i < options_.ignore_classes_size(); ++i) {
    ignore_classes_.insert(options_.ignore_classes(i));
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteTensorsToDetectionsCalculator::DecodeBoxes(
    const float* raw_boxes, const std::vector<Anchor>& anchors,
    std::vector<float>* boxes) {
  for (int i = 0; i < num_boxes_; ++i) {
    const int box_offset = i * num_coords_ + options_.box_coord_offset();

    float y_center = raw_boxes[box_offset];
    float x_center = raw_boxes[box_offset + 1];
    float h = raw_boxes[box_offset + 2];
    float w = raw_boxes[box_offset + 3];
    if (options_.reverse_output_order()) {
      x_center = raw_boxes[box_offset];
      y_center = raw_boxes[box_offset + 1];
      w = raw_boxes[box_offset + 2];
      h = raw_boxes[box_offset + 3];
    }

    x_center =
        x_center / options_.x_scale() * anchors[i].w() + anchors[i].x_center();
    y_center =
        y_center / options_.y_scale() * anchors[i].h() + anchors[i].y_center();

    if (options_.apply_exponential_on_box_size()) {
      h = std::exp(h / options_.h_scale()) * anchors[i].h();
      w = std::exp(w / options_.w_scale()) * anchors[i].w();
    } else {
      h = h / options_.h_scale() * anchors[i].h();
      w = w / options_.w_scale() * anchors[i].w();
    }

    const float ymin = y_center - h / 2.f;
    const float xmin = x_center - w / 2.f;
    const float ymax = y_center + h / 2.f;
    const float xmax = x_center + w / 2.f;

    (*boxes)[i * num_coords_ + 0] = ymin;
    (*boxes)[i * num_coords_ + 1] = xmin;
    (*boxes)[i * num_coords_ + 2] = ymax;
    (*boxes)[i * num_coords_ + 3] = xmax;

    if (options_.num_keypoints()) {
      for (int k = 0; k < options_.num_keypoints(); ++k) {
        const int offset = i * num_coords_ + options_.keypoint_coord_offset() +
                           k * options_.num_values_per_keypoint();

        float keypoint_y = raw_boxes[offset];
        float keypoint_x = raw_boxes[offset + 1];
        if (options_.reverse_output_order()) {
          keypoint_x = raw_boxes[offset];
          keypoint_y = raw_boxes[offset + 1];
        }

        (*boxes)[offset] = keypoint_x / options_.x_scale() * anchors[i].w() +
                           anchors[i].x_center();
        (*boxes)[offset + 1] =
            keypoint_y / options_.y_scale() * anchors[i].h() +
            anchors[i].y_center();
      }
    }
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteTensorsToDetectionsCalculator::ConvertToDetections(
    const float* detection_boxes, const float* detection_scores,
    const int* detection_classes, std::vector<Detection>* output_detections) {
  for (int i = 0; i < num_boxes_; ++i) {
    if (options_.has_min_score_thresh() &&
        detection_scores[i] < options_.min_score_thresh()) {
      continue;
    }
    const int box_offset = i * num_coords_;
    Detection detection = ConvertToDetection(
        detection_boxes[box_offset + 0], detection_boxes[box_offset + 1],
        detection_boxes[box_offset + 2], detection_boxes[box_offset + 3],
        detection_scores[i], detection_classes[i], options_.flip_vertically());
    // Add keypoints.
    if (options_.num_keypoints() > 0) {
      auto* location_data = detection.mutable_location_data();
      for (int kp_id = 0; kp_id < options_.num_keypoints() *
                                      options_.num_values_per_keypoint();
           kp_id += options_.num_values_per_keypoint()) {
        auto keypoint = location_data->add_relative_keypoints();
        const int keypoint_index =
            box_offset + options_.keypoint_coord_offset() + kp_id;
        keypoint->set_x(detection_boxes[keypoint_index + 0]);
        keypoint->set_y(options_.flip_vertically()
                            ? 1.f - detection_boxes[keypoint_index + 1]
                            : detection_boxes[keypoint_index + 1]);
      }
    }
    output_detections->emplace_back(detection);
  }
  return ::mediapipe::OkStatus();
}

Detection TfLiteTensorsToDetectionsCalculator::ConvertToDetection(
    float box_ymin, float box_xmin, float box_ymax, float box_xmax, float score,
    int class_id, bool flip_vertically) {
  Detection detection;
  detection.add_score(score);
  detection.add_label_id(class_id);

  LocationData* location_data = detection.mutable_location_data();
  location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);

  LocationData::RelativeBoundingBox* relative_bbox =
      location_data->mutable_relative_bounding_box();

  relative_bbox->set_xmin(box_xmin);
  relative_bbox->set_ymin(flip_vertically ? 1.f - box_ymax : box_ymin);
  relative_bbox->set_width(box_xmax - box_xmin);
  relative_bbox->set_height(box_ymax - box_ymin);
  return detection;
}

::mediapipe::Status TfLiteTensorsToDetectionsCalculator::GpuInit(
    CalculatorContext* cc) {
#if MEDIAPIPE_TFLITE_GL_INFERENCE
  MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this]()
                                                    -> ::mediapipe::Status {
    gpu_data_ = absl::make_unique<GPUData>();

    // A shader to decode detection boxes.
    const std::string decode_src = absl::Substitute(
        R"( #version 310 es

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(location = 0) uniform vec4 scale;

layout(std430, binding = 0) writeonly buffer Output {
  float data[];
} boxes;

layout(std430, binding = 1) readonly buffer Input0 {
  float data[];
} raw_boxes;

layout(std430, binding = 2) readonly buffer Input1 {
  float data[];
} raw_anchors;

uint num_coords = uint($0);
int reverse_output_order = int($1);
int apply_exponential = int($2);
int box_coord_offset = int($3);
int num_keypoints = int($4);
int keypt_coord_offset = int($5);
int num_values_per_keypt = int($6);

void main() {
  uint g_idx = gl_GlobalInvocationID.x;  // box index
  uint box_offset = g_idx * num_coords + uint(box_coord_offset);
  uint anchor_offset = g_idx * uint(4);  // check kNumCoordsPerBox

  float y_center, x_center, h, w;

  if (reverse_output_order == int(0)) {
    y_center = raw_boxes.data[box_offset + uint(0)];
    x_center = raw_boxes.data[box_offset + uint(1)];
    h = raw_boxes.data[box_offset + uint(2)];
    w = raw_boxes.data[box_offset + uint(3)];
  } else {
    x_center = raw_boxes.data[box_offset + uint(0)];
    y_center = raw_boxes.data[box_offset + uint(1)];
    w = raw_boxes.data[box_offset + uint(2)];
    h = raw_boxes.data[box_offset + uint(3)];
  }

  float anchor_yc = raw_anchors.data[anchor_offset + uint(0)];
  float anchor_xc = raw_anchors.data[anchor_offset + uint(1)];
  float anchor_h  = raw_anchors.data[anchor_offset + uint(2)];
  float anchor_w  = raw_anchors.data[anchor_offset + uint(3)];

  x_center = x_center / scale.x * anchor_w + anchor_xc;
  y_center = y_center / scale.y * anchor_h + anchor_yc;

  if (apply_exponential == int(1)) {
    h = exp(h / scale.w) * anchor_h;
    w = exp(w / scale.z) * anchor_w;
  } else {
    h = (h / scale.w) * anchor_h;
    w = (w / scale.z) * anchor_w;
  }

  float ymin = y_center - h / 2.0;
  float xmin = x_center - w / 2.0;
  float ymax = y_center + h / 2.0;
  float xmax = x_center + w / 2.0;

  boxes.data[box_offset + uint(0)] = ymin;
  boxes.data[box_offset + uint(1)] = xmin;
  boxes.data[box_offset + uint(2)] = ymax;
  boxes.data[box_offset + uint(3)] = xmax;

  if (num_keypoints > int(0)){
    for (int k = 0; k < num_keypoints; ++k) {
      int kp_offset =
        int(g_idx * num_coords) + keypt_coord_offset + k * num_values_per_keypt;
      float kp_y, kp_x;
      if (reverse_output_order == int(0)) {
        kp_y = raw_boxes.data[kp_offset + int(0)];
        kp_x = raw_boxes.data[kp_offset + int(1)];
      } else {
        kp_x = raw_boxes.data[kp_offset + int(0)];
        kp_y = raw_boxes.data[kp_offset + int(1)];
      }
      boxes.data[kp_offset + int(0)] = kp_x / scale.x * anchor_w + anchor_xc;
      boxes.data[kp_offset + int(1)] = kp_y / scale.y * anchor_h + anchor_yc;
    }
  }
})",
        options_.num_coords(),  // box xywh
        options_.reverse_output_order() ? 1 : 0,
        options_.apply_exponential_on_box_size() ? 1 : 0,
        options_.box_coord_offset(), options_.num_keypoints(),
        options_.keypoint_coord_offset(), options_.num_values_per_keypoint());

    // Shader program
    GlShader decode_shader;
    MP_RETURN_IF_ERROR(
        GlShader::CompileShader(GL_COMPUTE_SHADER, decode_src, &decode_shader));
    MP_RETURN_IF_ERROR(GpuProgram::CreateWithShader(
        decode_shader, &gpu_data_->decode_program));
    // Outputs
    size_t decoded_boxes_length = num_boxes_ * num_coords_;
    MP_RETURN_IF_ERROR(CreateReadWriteShaderStorageBuffer<float>(
        decoded_boxes_length, &gpu_data_->decoded_boxes_buffer));
    // Inputs
    size_t raw_boxes_length = num_boxes_ * num_coords_;
    MP_RETURN_IF_ERROR(CreateReadWriteShaderStorageBuffer<float>(
        raw_boxes_length, &gpu_data_->raw_boxes_buffer));
    size_t raw_anchors_length = num_boxes_ * kNumCoordsPerBox;
    MP_RETURN_IF_ERROR(CreateReadWriteShaderStorageBuffer<float>(
        raw_anchors_length, &gpu_data_->raw_anchors_buffer));
    // Parameters
    glUseProgram(gpu_data_->decode_program.id());
    glUniform4f(0, options_.x_scale(), options_.y_scale(), options_.w_scale(),
                options_.h_scale());

    // A shader to score detection boxes.
    const std::string score_src = absl::Substitute(
        R"( #version 310 es

layout(local_size_x = 1, local_size_y = $0, local_size_z = 1) in;

#define FLT_MAX 1.0e+37

shared float local_scores[$0];

layout(std430, binding = 0) writeonly buffer Output {
  float data[];
} scored_boxes;

layout(std430, binding = 1) readonly buffer Input0 {
  float data[];
} raw_scores;

uint num_classes = uint($0);
int apply_sigmoid = int($1);
int apply_clipping_thresh = int($2);
float clipping_thresh = float($3);
int ignore_class_0 = int($4);

float optional_sigmoid(float x) {
  if (apply_sigmoid == int(0)) return x;
  if (apply_clipping_thresh == int(1)) {
    x = clamp(x, -clipping_thresh, clipping_thresh);
  }
  x = 1.0 / (1.0 + exp(-x));
  return x;
}

void main() {
  uint g_idx = gl_GlobalInvocationID.x;   // box idx
  uint s_idx =  gl_LocalInvocationID.y;   // score/class idx

  // load all scores into shared memory
  float score = raw_scores.data[g_idx * num_classes + s_idx];
  local_scores[s_idx] = optional_sigmoid(score);
  memoryBarrierShared();
  barrier();

  // find max score in shared memory
  if (s_idx == uint(0)) {
    float max_score = -FLT_MAX;
    float max_class = -1.0;
    for (int i=ignore_class_0; i<int(num_classes); ++i) {
      if (local_scores[i] > max_score) {
        max_score = local_scores[i];
        max_class = float(i);
      }
    }
    scored_boxes.data[g_idx * uint(2) + uint(0)] = max_score;
    scored_boxes.data[g_idx * uint(2) + uint(1)] = max_class;
  }
})",
        num_classes_, options_.sigmoid_score() ? 1 : 0,
        options_.has_score_clipping_thresh() ? 1 : 0,
        options_.has_score_clipping_thresh() ? options_.score_clipping_thresh()
                                             : 0,
        !ignore_classes_.empty() ? 1 : 0);

    // # filter classes supported is hardware dependent.
    int max_wg_size;  //  typically <= 1024
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1,
                    &max_wg_size);  // y-dim
    CHECK_LT(num_classes_, max_wg_size)
        << "# classes must be < " << max_wg_size;
    // TODO support better filtering.
    CHECK_LE(ignore_classes_.size(), 1) << "Only ignore class 0 is allowed";

    // Shader program
    GlShader score_shader;
    MP_RETURN_IF_ERROR(
        GlShader::CompileShader(GL_COMPUTE_SHADER, score_src, &score_shader));
    MP_RETURN_IF_ERROR(
        GpuProgram::CreateWithShader(score_shader, &gpu_data_->score_program));
    // Outputs
    size_t scored_boxes_length = num_boxes_ * 2;  // score, class
    MP_RETURN_IF_ERROR(CreateReadWriteShaderStorageBuffer<float>(
        scored_boxes_length, &gpu_data_->scored_boxes_buffer));
    // Inputs
    size_t raw_scores_length = num_boxes_ * num_classes_;
    MP_RETURN_IF_ERROR(CreateReadWriteShaderStorageBuffer<float>(
        raw_scores_length, &gpu_data_->raw_scores_buffer));

    return ::mediapipe::OkStatus();
  }));

#elif MEDIAPIPE_TFLITE_METAL_INFERENCE

  gpu_data_ = absl::make_unique<GPUData>();
  id<MTLDevice> device = gpu_helper_.mtlDevice;

  // A shader to decode detection boxes.
  std::string decode_src = absl::Substitute(
      R"(
#include <metal_stdlib>

using namespace metal;

kernel void decodeKernel(
    device float*                   boxes       [[ buffer(0) ]],
    device float*                   raw_boxes   [[ buffer(1) ]],
    device float*                   raw_anchors [[ buffer(2) ]],
    uint2                           gid         [[ thread_position_in_grid ]]) {

  uint num_coords = uint($0);
  int reverse_output_order = int($1);
  int apply_exponential = int($2);
  int box_coord_offset = int($3);
  int num_keypoints = int($4);
  int keypt_coord_offset = int($5);
  int num_values_per_keypt = int($6);
)",
      options_.num_coords(),  // box xywh
      options_.reverse_output_order() ? 1 : 0,
      options_.apply_exponential_on_box_size() ? 1 : 0,
      options_.box_coord_offset(), options_.num_keypoints(),
      options_.keypoint_coord_offset(), options_.num_values_per_keypoint());
  decode_src += absl::Substitute(
      R"(
  float4 scale = float4(($0),($1),($2),($3));
)",
      options_.x_scale(), options_.y_scale(), options_.w_scale(),
      options_.h_scale());
  decode_src += R"(
  uint g_idx = gid.x;
  uint box_offset = g_idx * num_coords + uint(box_coord_offset);
  uint anchor_offset = g_idx * uint(4);  // check kNumCoordsPerBox

  float y_center, x_center, h, w;

  if (reverse_output_order == int(0)) {
    y_center = raw_boxes[box_offset + uint(0)];
    x_center = raw_boxes[box_offset + uint(1)];
    h = raw_boxes[box_offset + uint(2)];
    w = raw_boxes[box_offset + uint(3)];
  } else {
    x_center = raw_boxes[box_offset + uint(0)];
    y_center = raw_boxes[box_offset + uint(1)];
    w = raw_boxes[box_offset + uint(2)];
    h = raw_boxes[box_offset + uint(3)];
  }

  float anchor_yc = raw_anchors[anchor_offset + uint(0)];
  float anchor_xc = raw_anchors[anchor_offset + uint(1)];
  float anchor_h  = raw_anchors[anchor_offset + uint(2)];
  float anchor_w  = raw_anchors[anchor_offset + uint(3)];

  x_center = x_center / scale.x * anchor_w + anchor_xc;
  y_center = y_center / scale.y * anchor_h + anchor_yc;

  if (apply_exponential == int(1)) {
    h = exp(h / scale.w) * anchor_h;
    w = exp(w / scale.z) * anchor_w;
  } else {
    h = (h / scale.w) * anchor_h;
    w = (w / scale.z) * anchor_w;
  }

  float ymin = y_center - h / 2.0;
  float xmin = x_center - w / 2.0;
  float ymax = y_center + h / 2.0;
  float xmax = x_center + w / 2.0;

  boxes[box_offset + uint(0)] = ymin;
  boxes[box_offset + uint(1)] = xmin;
  boxes[box_offset + uint(2)] = ymax;
  boxes[box_offset + uint(3)] = xmax;

  if (num_keypoints > int(0)){
    for (int k = 0; k < num_keypoints; ++k) {
      int kp_offset =
        int(g_idx * num_coords) + keypt_coord_offset + k * num_values_per_keypt;
      float kp_y, kp_x;
      if (reverse_output_order == int(0)) {
        kp_y = raw_boxes[kp_offset + int(0)];
        kp_x = raw_boxes[kp_offset + int(1)];
      } else {
        kp_x = raw_boxes[kp_offset + int(0)];
        kp_y = raw_boxes[kp_offset + int(1)];
      }
      boxes[kp_offset + int(0)] = kp_x / scale.x * anchor_w + anchor_xc;
      boxes[kp_offset + int(1)] = kp_y / scale.y * anchor_h + anchor_yc;
    }
  }
})";

  {
    // Shader program
    NSString* library_source =
        [NSString stringWithUTF8String:decode_src.c_str()];
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:library_source
                                                  options:nullptr
                                                    error:&error];
    RET_CHECK(library != nil) << "Couldn't create shader library "
                              << [[error localizedDescription] UTF8String];
    id<MTLFunction> kernel_func = nil;
    kernel_func = [library newFunctionWithName:@"decodeKernel"];
    RET_CHECK(kernel_func != nil) << "Couldn't create kernel function.";
    gpu_data_->decode_program =
        [device newComputePipelineStateWithFunction:kernel_func error:&error];
    RET_CHECK(gpu_data_->decode_program != nil)
        << "Couldn't create pipeline state "
        << [[error localizedDescription] UTF8String];
    // Outputs
    size_t decoded_boxes_length = num_boxes_ * num_coords_ * sizeof(float);
    gpu_data_->decoded_boxes_buffer =
        [device newBufferWithLength:decoded_boxes_length
                            options:MTLResourceStorageModeShared];
    // Inputs
    size_t raw_boxes_length = num_boxes_ * num_coords_ * sizeof(float);
    gpu_data_->raw_boxes_buffer =
        [device newBufferWithLength:raw_boxes_length
                            options:MTLResourceStorageModeShared];
    size_t raw_anchors_length = num_boxes_ * kNumCoordsPerBox * sizeof(float);
    gpu_data_->raw_anchors_buffer =
        [device newBufferWithLength:raw_anchors_length
                            options:MTLResourceStorageModeShared];
  }

  // A shader to score detection boxes.
  const std::string score_src = absl::Substitute(
      R"(
#include <metal_stdlib>

using namespace metal;

float optional_sigmoid(float x) {
  int apply_sigmoid = int($1);
  int apply_clipping_thresh = int($2);
  float clipping_thresh = float($3);
  if (apply_sigmoid == int(0)) return x;
  if (apply_clipping_thresh == int(1)) {
    x = clamp(x, -clipping_thresh, clipping_thresh);
  }
  x = 1.0 / (1.0 + exp(-x));
  return x;
}

kernel void scoreKernel(
    device float*             scored_boxes [[ buffer(0) ]],
    device float*             raw_scores   [[ buffer(1) ]],
    uint2                     tid          [[ thread_position_in_threadgroup ]],
    uint2                     gid          [[ thread_position_in_grid ]]) {

  uint num_classes = uint($0);
  int apply_sigmoid = int($1);
  int apply_clipping_thresh = int($2);
  float clipping_thresh = float($3);
  int ignore_class_0 = int($4);

  uint g_idx = gid.x;   // box idx
  uint s_idx = tid.y;   // score/class idx

  // load all scores into shared memory
  threadgroup float local_scores[$0];
  float score = raw_scores[g_idx * num_classes + s_idx];
  local_scores[s_idx] = optional_sigmoid(score);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // find max score in shared memory
  if (s_idx == uint(0)) {
    float max_score = -FLT_MAX;
    float max_class = -1.0;
    for (int i=ignore_class_0; i<int(num_classes); ++i) {
      if (local_scores[i] > max_score) {
        max_score = local_scores[i];
        max_class = float(i);
      }
    }
    scored_boxes[g_idx * uint(2) + uint(0)] = max_score;
    scored_boxes[g_idx * uint(2) + uint(1)] = max_class;
  }
})",
      num_classes_, options_.sigmoid_score() ? 1 : 0,
      options_.has_score_clipping_thresh() ? 1 : 0,
      options_.has_score_clipping_thresh() ? options_.score_clipping_thresh()
                                           : 0,
      ignore_classes_.size() ? 1 : 0);

  // TODO support better filtering.
  CHECK_LE(ignore_classes_.size(), 1) << "Only ignore class 0 is allowed";

  {
    // Shader program
    NSString* library_source =
        [NSString stringWithUTF8String:score_src.c_str()];
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:library_source
                                                  options:nullptr
                                                    error:&error];
    RET_CHECK(library != nil) << "Couldn't create shader library "
                              << [[error localizedDescription] UTF8String];
    id<MTLFunction> kernel_func = nil;
    kernel_func = [library newFunctionWithName:@"scoreKernel"];
    RET_CHECK(kernel_func != nil) << "Couldn't create kernel function.";
    gpu_data_->score_program =
        [device newComputePipelineStateWithFunction:kernel_func error:&error];
    RET_CHECK(gpu_data_->score_program != nil)
        << "Couldn't create pipeline state "
        << [[error localizedDescription] UTF8String];
    // Outputs
    size_t scored_boxes_length = num_boxes_ * 2 * sizeof(float);  // score,class
    gpu_data_->scored_boxes_buffer =
        [device newBufferWithLength:scored_boxes_length
                            options:MTLResourceStorageModeShared];
    // Inputs
    size_t raw_scores_length = num_boxes_ * num_classes_ * sizeof(float);
    gpu_data_->raw_scores_buffer =
        [device newBufferWithLength:raw_scores_length
                            options:MTLResourceStorageModeShared];
    // # filter classes supported is hardware dependent.
    int max_wg_size = gpu_data_->score_program.maxTotalThreadsPerThreadgroup;
    CHECK_LT(num_classes_, max_wg_size) << "# classes must be <" << max_wg_size;
  }

#endif  // MEDIAPIPE_TFLITE_GL_INFERENCE

  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe

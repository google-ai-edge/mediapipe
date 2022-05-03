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
#include "mediapipe/calculators/tensor/tensors_to_detections_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location.h"
#include "mediapipe/framework/formats/object_detection/anchor.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/ret_check.h"

// Note: On Apple platforms MEDIAPIPE_DISABLE_GL_COMPUTE is automatically
// defined in mediapipe/framework/port.h. Therefore,
// "#ifndef MEDIAPIPE_DISABLE_GL_COMPUTE" and "#if MEDIAPIPE_METAL_ENABLED"
// below are mutually exclusive.
#ifndef MEDIAPIPE_DISABLE_GL_COMPUTE
#include "mediapipe/gpu/gl_calculator_helper.h"
#endif  // !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)

#if MEDIAPIPE_METAL_ENABLED
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#import "mediapipe/gpu/MPPMetalHelper.h"
#include "mediapipe/gpu/MPPMetalUtil.h"
#endif  // MEDIAPIPE_METAL_ENABLED

namespace {
constexpr int kNumInputTensorsWithAnchors = 3;
constexpr int kNumCoordsPerBox = 4;

bool CanUseGpu() {
#if !defined(MEDIAPIPE_DISABLE_GL_COMPUTE) || MEDIAPIPE_METAL_ENABLED
  // TODO: Configure GPU usage policy in individual calculators.
  constexpr bool kAllowGpuProcessing = true;
  return kAllowGpuProcessing;
#else
  return false;
#endif  // !defined(MEDIAPIPE_DISABLE_GL_COMPUTE) || MEDIAPIPE_METAL_ENABLED
}
}  // namespace

namespace mediapipe {
namespace api2 {

namespace {

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

absl::Status CheckCustomTensorMapping(
    const TensorsToDetectionsCalculatorOptions::TensorMapping& tensor_mapping) {
  RET_CHECK(tensor_mapping.has_detections_tensor_index() &&
            tensor_mapping.has_scores_tensor_index());
  int bitmap = 0;
  bitmap |= 1 << tensor_mapping.detections_tensor_index();
  bitmap |= 1 << tensor_mapping.scores_tensor_index();
  if (!tensor_mapping.has_num_detections_tensor_index() &&
      !tensor_mapping.has_classes_tensor_index() &&
      !tensor_mapping.has_anchors_tensor_index()) {
    // Only allows the output tensor index 0 and 1 to be occupied.
    RET_CHECK_EQ(3, bitmap) << "The custom output tensor indices should only "
                               "cover index 0 and 1.";
  } else if (tensor_mapping.has_anchors_tensor_index()) {
    RET_CHECK(!tensor_mapping.has_classes_tensor_index() &&
              !tensor_mapping.has_num_detections_tensor_index());
    bitmap |= 1 << tensor_mapping.anchors_tensor_index();
    // If the"anchors" tensor will be available, only allows the output tensor
    // index 0, 1, 2 to be occupied.
    RET_CHECK_EQ(7, bitmap) << "The custom output tensor indices should only "
                               "cover index 0, 1 and 2.";
  } else {
    RET_CHECK(tensor_mapping.has_classes_tensor_index() &&
              tensor_mapping.has_num_detections_tensor_index());
    // If the "classes" and the "number of detections" tensors will be
    // available, only allows the output tensor index 0, 1, 2, 3 to be occupied.
    bitmap |= 1 << tensor_mapping.classes_tensor_index();
    bitmap |= 1 << tensor_mapping.num_detections_tensor_index();
    RET_CHECK_EQ(15, bitmap) << "The custom output tensor indices should only "
                                "cover index 0, 1, 2 and 3.";
  }
  return absl::OkStatus();
}

}  // namespace

// Convert result Tensors from object detection models into MediaPipe
// Detections.
//
// Input:
//  TENSORS - Vector of Tensors of type kFloat32. The vector of tensors can have
//            2 or 3 tensors. First tensor is the predicted raw boxes/keypoints.
//            The size of the values must be (num_boxes * num_predicted_values).
//            Second tensor is the score tensor. The size of the valuse must be
//            (num_boxes * num_classes). It's optional to pass in a third tensor
//            for anchors (e.g. for SSD models) depend on the outputs of the
//            detection model. The size of anchor tensor must be (num_boxes *
//            4).
//
// Input side packet:
//  ANCHORS (optional) - The anchors used for decoding the bounding boxes, as a
//      vector of `Anchor` protos. Not required if post-processing is built-in
//      the model.
//  IGNORE_CLASSES (optional) - The list of class ids that should be ignored, as
//      a vector of integers. It overrides the corresponding field in the
//      calculator options.
//
// Output:
//  DETECTIONS - Result MediaPipe detections.
//
// Usage example:
// node {
//   calculator: "TensorsToDetectionsCalculator"
//   input_stream: "TENSORS:tensors"
//   input_side_packet: "ANCHORS:anchors"
//   output_stream: "DETECTIONS:detections"
//   options: {
//     [mediapipe.TensorsToDetectionsCalculatorOptions.ext] {
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
class TensorsToDetectionsCalculator : public Node {
 public:
  static constexpr Input<std::vector<Tensor>> kInTensors{"TENSORS"};
  static constexpr SideInput<std::vector<Anchor>>::Optional kInAnchors{
      "ANCHORS"};
  static constexpr SideInput<std::vector<int>>::Optional kSideInIgnoreClasses{
      "IGNORE_CLASSES"};
  static constexpr Output<std::vector<Detection>> kOutDetections{"DETECTIONS"};
  MEDIAPIPE_NODE_CONTRACT(kInTensors, kInAnchors, kSideInIgnoreClasses,
                          kOutDetections);
  static absl::Status UpdateContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status ProcessCPU(CalculatorContext* cc,
                          std::vector<Detection>* output_detections);
  absl::Status ProcessGPU(CalculatorContext* cc,
                          std::vector<Detection>* output_detections);

  absl::Status LoadOptions(CalculatorContext* cc);
  absl::Status GpuInit(CalculatorContext* cc);
  absl::Status DecodeBoxes(const float* raw_boxes,
                           const std::vector<Anchor>& anchors,
                           std::vector<float>* boxes);
  absl::Status ConvertToDetections(const float* detection_boxes,
                                   const float* detection_scores,
                                   const int* detection_classes,
                                   std::vector<Detection>* output_detections);
  Detection ConvertToDetection(float box_ymin, float box_xmin, float box_ymax,
                               float box_xmax, float score, int class_id,
                               bool flip_vertically);
  bool IsClassIndexAllowed(int class_index);

  int num_classes_ = 0;
  int num_boxes_ = 0;
  int num_coords_ = 0;
  int max_results_ = -1;

  // Set of allowed or ignored class indices.
  struct ClassIndexSet {
    absl::flat_hash_set<int> values;
    bool is_allowlist;
  };
  // Allowed or ignored class indices based on provided options or side packet.
  // These are used to filter out the output detection results.
  ClassIndexSet class_index_set_;

  TensorsToDetectionsCalculatorOptions options_;
  bool scores_tensor_index_is_set_ = false;
  TensorsToDetectionsCalculatorOptions::TensorMapping tensor_mapping_;
  std::vector<int> box_indices_ = {0, 1, 2, 3};
  bool has_custom_box_indices_ = false;
  std::vector<Anchor> anchors_;

#ifndef MEDIAPIPE_DISABLE_GL_COMPUTE
  mediapipe::GlCalculatorHelper gpu_helper_;
  GLuint decode_program_;
  GLuint score_program_;
#elif MEDIAPIPE_METAL_ENABLED
  MPPMetalHelper* gpu_helper_ = nullptr;
  id<MTLComputePipelineState> decode_program_;
  id<MTLComputePipelineState> score_program_;
#endif  // !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
  std::unique_ptr<Tensor> raw_anchors_buffer_;
  std::unique_ptr<Tensor> decoded_boxes_buffer_;
  std::unique_ptr<Tensor> scored_boxes_buffer_;

  bool gpu_inited_ = false;
  bool gpu_input_ = false;
  bool anchors_init_ = false;
};
MEDIAPIPE_REGISTER_NODE(TensorsToDetectionsCalculator);

absl::Status TensorsToDetectionsCalculator::UpdateContract(
    CalculatorContract* cc) {
  if (CanUseGpu()) {
#ifndef MEDIAPIPE_DISABLE_GL_COMPUTE
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#elif MEDIAPIPE_METAL_ENABLED
    MP_RETURN_IF_ERROR([MPPMetalHelper updateContract:cc]);
#endif  // !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
  }

  return absl::OkStatus();
}

absl::Status TensorsToDetectionsCalculator::Open(CalculatorContext* cc) {
  MP_RETURN_IF_ERROR(LoadOptions(cc));

  if (CanUseGpu()) {
#ifndef MEDIAPIPE_DISABLE_GL_COMPUTE
    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
#elif MEDIAPIPE_METAL_ENABLED
    gpu_helper_ = [[MPPMetalHelper alloc] initWithCalculatorContext:cc];
    RET_CHECK(gpu_helper_);
#endif  // !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
  }

  return absl::OkStatus();
}

absl::Status TensorsToDetectionsCalculator::Process(CalculatorContext* cc) {
  auto output_detections = absl::make_unique<std::vector<Detection>>();
  bool gpu_processing = false;
  if (CanUseGpu()) {
    // Use GPU processing only if at least one input tensor is already on GPU
    // (to avoid CPU->GPU overhead).
    for (const auto& tensor : *kInTensors(cc)) {
      if (tensor.ready_on_gpu()) {
        gpu_processing = true;
        break;
      }
    }
  }
  const int num_input_tensors = kInTensors(cc)->size();
  if (!scores_tensor_index_is_set_) {
    if (num_input_tensors == 2 ||
        num_input_tensors == kNumInputTensorsWithAnchors) {
      tensor_mapping_.set_scores_tensor_index(1);
    } else {
      tensor_mapping_.set_scores_tensor_index(2);
    }
    scores_tensor_index_is_set_ = true;
  }
  if (gpu_processing || num_input_tensors != 4) {
    // Allows custom bounding box indices when receiving 4 cpu tensors.
    // Uses the default bbox indices in other cases.
    RET_CHECK(!has_custom_box_indices_);
  }

  if (gpu_processing) {
    if (!gpu_inited_) {
      MP_RETURN_IF_ERROR(GpuInit(cc));
      gpu_inited_ = true;
    }
    MP_RETURN_IF_ERROR(ProcessGPU(cc, output_detections.get()));
  } else {
    MP_RETURN_IF_ERROR(ProcessCPU(cc, output_detections.get()));
  }

  kOutDetections(cc).Send(std::move(output_detections));
  return absl::OkStatus();
}

absl::Status TensorsToDetectionsCalculator::ProcessCPU(
    CalculatorContext* cc, std::vector<Detection>* output_detections) {
  const auto& input_tensors = *kInTensors(cc);

  if (input_tensors.size() == 2 ||
      input_tensors.size() == kNumInputTensorsWithAnchors) {
    // Postprocessing on CPU for model without postprocessing op. E.g. output
    // raw score tensor and box tensor. Anchor decoding will be handled below.
    // TODO: Add flexible input tensor size handling.
    auto raw_box_tensor =
        &input_tensors[tensor_mapping_.detections_tensor_index()];
    RET_CHECK_EQ(raw_box_tensor->shape().dims.size(), 3);
    RET_CHECK_EQ(raw_box_tensor->shape().dims[0], 1);
    RET_CHECK_GT(num_boxes_, 0) << "Please set num_boxes in calculator options";
    RET_CHECK_EQ(raw_box_tensor->shape().dims[1], num_boxes_);
    RET_CHECK_EQ(raw_box_tensor->shape().dims[2], num_coords_);
    auto raw_score_tensor =
        &input_tensors[tensor_mapping_.scores_tensor_index()];
    RET_CHECK_EQ(raw_score_tensor->shape().dims.size(), 3);
    RET_CHECK_EQ(raw_score_tensor->shape().dims[0], 1);
    RET_CHECK_EQ(raw_score_tensor->shape().dims[1], num_boxes_);
    RET_CHECK_EQ(raw_score_tensor->shape().dims[2], num_classes_);
    auto raw_box_view = raw_box_tensor->GetCpuReadView();
    auto raw_boxes = raw_box_view.buffer<float>();
    auto raw_scores_view = raw_score_tensor->GetCpuReadView();
    auto raw_scores = raw_scores_view.buffer<float>();

    // TODO: Support other options to load anchors.
    if (!anchors_init_) {
      if (input_tensors.size() == kNumInputTensorsWithAnchors) {
        auto anchor_tensor =
            &input_tensors[tensor_mapping_.anchors_tensor_index()];
        RET_CHECK_EQ(anchor_tensor->shape().dims.size(), 2);
        RET_CHECK_EQ(anchor_tensor->shape().dims[0], num_boxes_);
        RET_CHECK_EQ(anchor_tensor->shape().dims[1], kNumCoordsPerBox);
        auto anchor_view = anchor_tensor->GetCpuReadView();
        auto raw_anchors = anchor_view.buffer<float>();
        ConvertRawValuesToAnchors(raw_anchors, num_boxes_, &anchors_);
      } else if (!kInAnchors(cc).IsEmpty()) {
        anchors_ = *kInAnchors(cc);
      } else {
        return absl::UnavailableError("No anchor data available.");
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
        if (IsClassIndexAllowed(score_idx)) {
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
    auto num_boxes_tensor =
        &input_tensors[tensor_mapping_.num_detections_tensor_index()];
    RET_CHECK_EQ(num_boxes_tensor->shape().dims.size(), 1);
    RET_CHECK_EQ(num_boxes_tensor->shape().dims[0], 1);

    auto detection_boxes_tensor =
        &input_tensors[tensor_mapping_.detections_tensor_index()];
    RET_CHECK_EQ(detection_boxes_tensor->shape().dims.size(), 3);
    RET_CHECK_EQ(detection_boxes_tensor->shape().dims[0], 1);
    const int max_detections = detection_boxes_tensor->shape().dims[1];
    RET_CHECK_EQ(detection_boxes_tensor->shape().dims[2], num_coords_);

    auto detection_classes_tensor =
        &input_tensors[tensor_mapping_.classes_tensor_index()];
    RET_CHECK_EQ(detection_classes_tensor->shape().dims.size(), 2);
    RET_CHECK_EQ(detection_classes_tensor->shape().dims[0], 1);
    RET_CHECK_EQ(detection_classes_tensor->shape().dims[1], max_detections);

    auto detection_scores_tensor =
        &input_tensors[tensor_mapping_.scores_tensor_index()];
    RET_CHECK_EQ(detection_scores_tensor->shape().dims.size(), 2);
    RET_CHECK_EQ(detection_scores_tensor->shape().dims[0], 1);
    RET_CHECK_EQ(detection_scores_tensor->shape().dims[1], max_detections);

    auto num_boxes_view = num_boxes_tensor->GetCpuReadView();
    auto num_boxes = num_boxes_view.buffer<float>();
    num_boxes_ = num_boxes[0];

    auto detection_boxes_view = detection_boxes_tensor->GetCpuReadView();
    auto detection_boxes = detection_boxes_view.buffer<float>();

    auto detection_scores_view = detection_scores_tensor->GetCpuReadView();
    auto detection_scores = detection_scores_view.buffer<float>();

    auto detection_classes_view = detection_classes_tensor->GetCpuReadView();
    auto detection_classes_ptr = detection_classes_view.buffer<float>();
    std::vector<int> detection_classes(num_boxes_);
    for (int i = 0; i < num_boxes_; ++i) {
      detection_classes[i] = static_cast<int>(detection_classes_ptr[i]);
    }
    MP_RETURN_IF_ERROR(ConvertToDetections(detection_boxes, detection_scores,
                                           detection_classes.data(),
                                           output_detections));
  }
  return absl::OkStatus();
}

absl::Status TensorsToDetectionsCalculator::ProcessGPU(
    CalculatorContext* cc, std::vector<Detection>* output_detections) {
  const auto& input_tensors = *kInTensors(cc);
  RET_CHECK_GE(input_tensors.size(), 2);
  RET_CHECK_GT(num_boxes_, 0) << "Please set num_boxes in calculator options";
#ifndef MEDIAPIPE_DISABLE_GL_COMPUTE

  MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this, &input_tensors, &cc,
                                                 &output_detections]()
                                                    -> absl::Status {
    if (!anchors_init_) {
      if (input_tensors.size() == kNumInputTensorsWithAnchors) {
        auto read_view = input_tensors[tensor_mapping_.anchors_tensor_index()]
                             .GetOpenGlBufferReadView();
        glBindBuffer(GL_COPY_READ_BUFFER, read_view.name());
        auto write_view = raw_anchors_buffer_->GetOpenGlBufferWriteView();
        glBindBuffer(GL_COPY_WRITE_BUFFER, write_view.name());
        glCopyBufferSubData(
            GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0,
            input_tensors[tensor_mapping_.anchors_tensor_index()].bytes());
      } else if (!kInAnchors(cc).IsEmpty()) {
        const auto& anchors = *kInAnchors(cc);
        auto anchors_view = raw_anchors_buffer_->GetCpuWriteView();
        auto raw_anchors = anchors_view.buffer<float>();
        ConvertAnchorsToRawValues(anchors, num_boxes_, raw_anchors);
      } else {
        return absl::UnavailableError("No anchor data available.");
      }
      anchors_init_ = true;
    }
    // Use the scope to release the writable buffers' views before requesting
    // the reading buffers' views.
    {
      // Decode boxes.
      auto scored_boxes_view = scored_boxes_buffer_->GetOpenGlBufferWriteView();
      auto decoded_boxes_view =
          decoded_boxes_buffer_->GetOpenGlBufferWriteView();
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, decoded_boxes_view.name());
      auto input0_view =
          input_tensors[tensor_mapping_.detections_tensor_index()]
              .GetOpenGlBufferReadView();
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, input0_view.name());
      auto raw_anchors_view = raw_anchors_buffer_->GetOpenGlBufferReadView();
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, raw_anchors_view.name());
      glUseProgram(decode_program_);
      glDispatchCompute(num_boxes_, 1, 1);

      // Score boxes.
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, scored_boxes_view.name());
      auto input1_view = input_tensors[tensor_mapping_.scores_tensor_index()]
                             .GetOpenGlBufferReadView();
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, input1_view.name());
      glUseProgram(score_program_);
      glDispatchCompute(num_boxes_, 1, 1);
    }
    return absl::OkStatus();
  }));

  // TODO: b/138851969. Is it possible to output a float vector
  // for score and an int vector for class so that we can avoid copying twice?
  std::vector<float> detection_scores(num_boxes_);
  std::vector<int> detection_classes(num_boxes_);
  // The order of requesting of CpuViews must be the same as the order of
  // requesting OpenGlViews above to avoid 'Potential mutex deadlock' message
  // when compiled without '-c opt' option.
  auto scored_boxes_view = scored_boxes_buffer_->GetCpuReadView();
  auto score_class_id_pairs = scored_boxes_view.buffer<float>();
  for (int i = 0; i < num_boxes_; ++i) {
    detection_scores[i] = score_class_id_pairs[i * 2];
    detection_classes[i] = static_cast<int>(score_class_id_pairs[i * 2 + 1]);
  }
  auto decoded_boxes_view = decoded_boxes_buffer_->GetCpuReadView();
  auto boxes = decoded_boxes_view.buffer<float>();
  MP_RETURN_IF_ERROR(ConvertToDetections(boxes, detection_scores.data(),
                                         detection_classes.data(),
                                         output_detections));
#elif MEDIAPIPE_METAL_ENABLED
  id<MTLDevice> device = gpu_helper_.mtlDevice;
  if (!anchors_init_) {
    if (input_tensors.size() == kNumInputTensorsWithAnchors) {
      RET_CHECK_EQ(input_tensors.size(), kNumInputTensorsWithAnchors);
      auto command_buffer = [gpu_helper_ commandBuffer];
      auto src_buffer = input_tensors[tensor_mapping_.anchors_tensor_index()]
                            .GetMtlBufferReadView(command_buffer);
      auto dest_buffer =
          raw_anchors_buffer_->GetMtlBufferWriteView(command_buffer);
      id<MTLBlitCommandEncoder> blit_command =
          [command_buffer blitCommandEncoder];
      [blit_command copyFromBuffer:src_buffer.buffer()
                      sourceOffset:0
                          toBuffer:dest_buffer.buffer()
                 destinationOffset:0
                              size:input_tensors[tensor_mapping_
                                                     .anchors_tensor_index()]
                                       .bytes()];
      [blit_command endEncoding];
      [command_buffer commit];
    } else if (!kInAnchors(cc).IsEmpty()) {
      const auto& anchors = *kInAnchors(cc);
      auto raw_anchors_view = raw_anchors_buffer_->GetCpuWriteView();
      ConvertAnchorsToRawValues(anchors, num_boxes_,
                                raw_anchors_view.buffer<float>());
    } else {
      return absl::UnavailableError("No anchor data available.");
    }
    anchors_init_ = true;
  }

  // Use the scope to release the writable buffers' views before requesting the
  // reading buffers' views.
  id<MTLCommandBuffer> command_buffer = [gpu_helper_ commandBuffer];
  command_buffer.label = @"DecodeAndScoreBoxes";
  id<MTLComputeCommandEncoder> command_encoder =
      [command_buffer computeCommandEncoder];
  [command_encoder setComputePipelineState:decode_program_];
  {
    auto scored_boxes_view =
        scored_boxes_buffer_->GetMtlBufferWriteView(command_buffer);
    auto decoded_boxes_view =
        decoded_boxes_buffer_->GetMtlBufferWriteView(command_buffer);
    [command_encoder setBuffer:decoded_boxes_view.buffer() offset:0 atIndex:0];
    auto input0_view = input_tensors[tensor_mapping_.detections_tensor_index()]
                           .GetMtlBufferReadView(command_buffer);
    [command_encoder setBuffer:input0_view.buffer() offset:0 atIndex:1];
    auto raw_anchors_view =
        raw_anchors_buffer_->GetMtlBufferReadView(command_buffer);
    [command_encoder setBuffer:raw_anchors_view.buffer() offset:0 atIndex:2];
    MTLSize decode_threads_per_group = MTLSizeMake(1, 1, 1);
    MTLSize decode_threadgroups = MTLSizeMake(num_boxes_, 1, 1);
    [command_encoder dispatchThreadgroups:decode_threadgroups
                    threadsPerThreadgroup:decode_threads_per_group];

    [command_encoder setComputePipelineState:score_program_];
    [command_encoder setBuffer:scored_boxes_view.buffer() offset:0 atIndex:0];
    auto input1_view = input_tensors[tensor_mapping_.scores_tensor_index()]
                           .GetMtlBufferReadView(command_buffer);
    [command_encoder setBuffer:input1_view.buffer() offset:0 atIndex:1];
    MTLSize score_threads_per_group = MTLSizeMake(1, num_classes_, 1);
    MTLSize score_threadgroups = MTLSizeMake(num_boxes_, 1, 1);
    [command_encoder dispatchThreadgroups:score_threadgroups
                    threadsPerThreadgroup:score_threads_per_group];
    [command_encoder endEncoding];
    [command_buffer commit];
  }

  // Output detections.
  // TODO Adjust shader to avoid copying shader output twice.
  std::vector<float> detection_scores(num_boxes_);
  std::vector<int> detection_classes(num_boxes_);
  {
    auto scored_boxes_view = scored_boxes_buffer_->GetCpuReadView();
    auto score_class_id_pairs = scored_boxes_view.buffer<float>();
    for (int i = 0; i < num_boxes_; ++i) {
      detection_scores[i] = score_class_id_pairs[i * 2];
      detection_classes[i] = static_cast<int>(score_class_id_pairs[i * 2 + 1]);
    }
  }
  auto decoded_boxes_view = decoded_boxes_buffer_->GetCpuReadView();
  auto boxes = decoded_boxes_view.buffer<float>();
  MP_RETURN_IF_ERROR(ConvertToDetections(boxes, detection_scores.data(),
                                         detection_classes.data(),
                                         output_detections));

#else
  LOG(ERROR) << "GPU input on non-Android not supported yet.";
#endif  // !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)
  return absl::OkStatus();
}

absl::Status TensorsToDetectionsCalculator::Close(CalculatorContext* cc) {
#ifndef MEDIAPIPE_DISABLE_GL_COMPUTE
  gpu_helper_.RunInGlContext([this] {
    decoded_boxes_buffer_ = nullptr;
    scored_boxes_buffer_ = nullptr;
    raw_anchors_buffer_ = nullptr;
    glDeleteProgram(decode_program_);
    glDeleteProgram(score_program_);
  });
#elif MEDIAPIPE_METAL_ENABLED
  decoded_boxes_buffer_ = nullptr;
  scored_boxes_buffer_ = nullptr;
  raw_anchors_buffer_ = nullptr;
  decode_program_ = nil;
  score_program_ = nil;
#endif  // !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)

  return absl::OkStatus();
}

absl::Status TensorsToDetectionsCalculator::LoadOptions(CalculatorContext* cc) {
  // Get calculator options specified in the graph.
  options_ = cc->Options<::mediapipe::TensorsToDetectionsCalculatorOptions>();
  RET_CHECK(options_.has_num_classes());
  RET_CHECK(options_.has_num_coords());

  num_classes_ = options_.num_classes();
  num_boxes_ = options_.num_boxes();
  num_coords_ = options_.num_coords();
  CHECK_NE(options_.max_results(), 0)
      << "The maximum number of the top-scored detection results must be "
         "non-zero.";
  max_results_ = options_.max_results();

  // Currently only support 2D when num_values_per_keypoint equals to 2.
  CHECK_EQ(options_.num_values_per_keypoint(), 2);

  // Check if the output size is equal to the requested boxes and keypoints.
  CHECK_EQ(options_.num_keypoints() * options_.num_values_per_keypoint() +
               kNumCoordsPerBox,
           num_coords_);

  if (kSideInIgnoreClasses(cc).IsConnected()) {
    RET_CHECK(!kSideInIgnoreClasses(cc).IsEmpty());
    RET_CHECK(options_.allow_classes().empty());
    class_index_set_.is_allowlist = false;
    for (int ignore_class : *kSideInIgnoreClasses(cc)) {
      class_index_set_.values.insert(ignore_class);
    }
  } else if (!options_.allow_classes().empty()) {
    RET_CHECK(options_.ignore_classes().empty());
    class_index_set_.is_allowlist = true;
    for (int i = 0; i < options_.allow_classes_size(); ++i) {
      class_index_set_.values.insert(options_.allow_classes(i));
    }
  } else {
    class_index_set_.is_allowlist = false;
    for (int i = 0; i < options_.ignore_classes_size(); ++i) {
      class_index_set_.values.insert(options_.ignore_classes(i));
    }
  }

  if (options_.has_tensor_mapping()) {
    RET_CHECK_OK(CheckCustomTensorMapping(options_.tensor_mapping()));
    tensor_mapping_ = options_.tensor_mapping();
    scores_tensor_index_is_set_ = true;
  } else {
    // Assigns the default tensor indices.
    tensor_mapping_.set_detections_tensor_index(0);
    tensor_mapping_.set_classes_tensor_index(1);
    tensor_mapping_.set_anchors_tensor_index(2);
    tensor_mapping_.set_num_detections_tensor_index(3);
    // The scores tensor index needs to be determined based on the number of
    // model's output tensors, which will be available in the first invocation
    // of the Process() method.
    tensor_mapping_.set_scores_tensor_index(-1);
    scores_tensor_index_is_set_ = false;
  }

  if (options_.has_box_boundaries_indices()) {
    box_indices_ = {options_.box_boundaries_indices().ymin(),
                    options_.box_boundaries_indices().xmin(),
                    options_.box_boundaries_indices().ymax(),
                    options_.box_boundaries_indices().xmax()};
    int bitmap = 0;
    for (int i : box_indices_) {
      bitmap |= 1 << i;
    }
    RET_CHECK_EQ(bitmap, 15) << "The custom box boundaries indices should only "
                                "cover index 0, 1, 2, and 3.";
    has_custom_box_indices_ = true;
  }

  return absl::OkStatus();
}

absl::Status TensorsToDetectionsCalculator::DecodeBoxes(
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

  return absl::OkStatus();
}

absl::Status TensorsToDetectionsCalculator::ConvertToDetections(
    const float* detection_boxes, const float* detection_scores,
    const int* detection_classes, std::vector<Detection>* output_detections) {
  for (int i = 0; i < num_boxes_; ++i) {
    if (max_results_ > 0 && output_detections->size() == max_results_) {
      break;
    }
    if (options_.has_min_score_thresh() &&
        detection_scores[i] < options_.min_score_thresh()) {
      continue;
    }
    if (!IsClassIndexAllowed(detection_classes[i])) {
      continue;
    }
    const int box_offset = i * num_coords_;
    Detection detection = ConvertToDetection(
        /*box_ymin=*/detection_boxes[box_offset + box_indices_[0]],
        /*box_xmin=*/detection_boxes[box_offset + box_indices_[1]],
        /*box_ymax=*/detection_boxes[box_offset + box_indices_[2]],
        /*box_xmax=*/detection_boxes[box_offset + box_indices_[3]],
        detection_scores[i], detection_classes[i], options_.flip_vertically());
    const auto& bbox = detection.location_data().relative_bounding_box();
    if (bbox.width() < 0 || bbox.height() < 0 || std::isnan(bbox.width()) ||
        std::isnan(bbox.height())) {
      // Decoded detection boxes could have negative values for width/height due
      // to model prediction. Filter out those boxes since some downstream
      // calculators may assume non-negative values. (b/171391719)
      continue;
    }
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
  return absl::OkStatus();
}

Detection TensorsToDetectionsCalculator::ConvertToDetection(
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

absl::Status TensorsToDetectionsCalculator::GpuInit(CalculatorContext* cc) {
#ifndef MEDIAPIPE_DISABLE_GL_COMPUTE
  MP_RETURN_IF_ERROR(gpu_helper_.RunInGlContext([this]() -> absl::Status {
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
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    const GLchar* sources[] = {decode_src.c_str()};
    glShaderSource(shader, 1, sources, NULL);
    glCompileShader(shader);
    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    RET_CHECK(compiled == GL_TRUE) << "Shader compilation error: " << [shader] {
      GLint length;
      glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
      std::string str;
      str.reserve(length);
      glGetShaderInfoLog(shader, length, nullptr, str.data());
      return str;
    }();
    decode_program_ = glCreateProgram();
    glAttachShader(decode_program_, shader);
    glDeleteShader(shader);
    glLinkProgram(decode_program_);

    // Outputs
    decoded_boxes_buffer_ =
        absl::make_unique<Tensor>(Tensor::ElementType::kFloat32,
                                  Tensor::Shape{1, num_boxes_ * num_coords_});
    raw_anchors_buffer_ = absl::make_unique<Tensor>(
        Tensor::ElementType::kFloat32,
        Tensor::Shape{1, num_boxes_ * kNumCoordsPerBox});
    // Parameters
    glUseProgram(decode_program_);
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
        !IsClassIndexAllowed(0));

    // # filter classes supported is hardware dependent.
    int max_wg_size;  //  typically <= 1024
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1,
                    &max_wg_size);  // y-dim
    CHECK_LT(num_classes_, max_wg_size)
        << "# classes must be < " << max_wg_size;
    // TODO support better filtering.
    if (class_index_set_.is_allowlist) {
      CHECK_EQ(class_index_set_.values.size(),
               IsClassIndexAllowed(0) ? num_classes_ : num_classes_ - 1)
          << "Only all classes  >= class 0  or  >= class 1";
    } else {
      CHECK_EQ(class_index_set_.values.size(), IsClassIndexAllowed(0) ? 0 : 1)
          << "Only ignore class 0 is allowed";
    }

    // Shader program
    {
      GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
      const GLchar* sources[] = {score_src.c_str()};
      glShaderSource(shader, 1, sources, NULL);
      glCompileShader(shader);
      GLint compiled = GL_FALSE;
      glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
      RET_CHECK(compiled == GL_TRUE);
      score_program_ = glCreateProgram();
      glAttachShader(score_program_, shader);
      glDeleteShader(shader);
      glLinkProgram(score_program_);
    }

    // Outputs
    scored_boxes_buffer_ = absl::make_unique<Tensor>(
        Tensor::ElementType::kFloat32, Tensor::Shape{1, num_boxes_ * 2});

    return absl::OkStatus();
  }));

#elif MEDIAPIPE_METAL_ENABLED
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
    decode_program_ =
        [device newComputePipelineStateWithFunction:kernel_func error:&error];
    RET_CHECK(decode_program_ != nil) << "Couldn't create pipeline state " <<
        [[error localizedDescription] UTF8String];
    // Outputs
    decoded_boxes_buffer_ =
        absl::make_unique<Tensor>(Tensor::ElementType::kFloat32,
                                  Tensor::Shape{1, num_boxes_ * num_coords_});
    // Inputs
    raw_anchors_buffer_ = absl::make_unique<Tensor>(
        Tensor::ElementType::kFloat32,
        Tensor::Shape{1, num_boxes_ * kNumCoordsPerBox});
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
      !IsClassIndexAllowed(0));

  // TODO support better filtering.
  if (class_index_set_.is_allowlist) {
    CHECK_EQ(class_index_set_.values.size(),
             IsClassIndexAllowed(0) ? num_classes_ : num_classes_ - 1)
        << "Only all classes  >= class 0  or  >= class 1";
  } else {
    CHECK_EQ(class_index_set_.values.size(), IsClassIndexAllowed(0) ? 0 : 1)
        << "Only ignore class 0 is allowed";
  }

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
    score_program_ =
        [device newComputePipelineStateWithFunction:kernel_func error:&error];
    RET_CHECK(score_program_ != nil) << "Couldn't create pipeline state " <<
        [[error localizedDescription] UTF8String];
    // Outputs
    scored_boxes_buffer_ = absl::make_unique<Tensor>(
        Tensor::ElementType::kFloat32, Tensor::Shape{1, num_boxes_ * 2});
    // # filter classes supported is hardware dependent.
    int max_wg_size = score_program_.maxTotalThreadsPerThreadgroup;
    CHECK_LT(num_classes_, max_wg_size) << "# classes must be <" << max_wg_size;
  }

#endif  // !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)

  return absl::OkStatus();
}

bool TensorsToDetectionsCalculator::IsClassIndexAllowed(int class_index) {
  if (class_index_set_.values.empty()) {
    return true;
  }
  if (class_index_set_.is_allowlist) {
    return class_index_set_.values.contains(class_index);
  } else {
    return !class_index_set_.values.contains(class_index);
  }
}

}  // namespace api2
}  // namespace mediapipe

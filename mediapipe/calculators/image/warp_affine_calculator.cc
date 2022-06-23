// Copyright 2021 The MediaPipe Authors.
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

#include "mediapipe/calculators/image/warp_affine_calculator.h"

#include <array>
#include <cstdint>
#include <memory>

#include "mediapipe/calculators/image/affine_transformation.h"
#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/calculators/image/affine_transformation_runner_gl.h"
#endif  // !MEDIAPIPE_DISABLE_GPU
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#if !MEDIAPIPE_DISABLE_OPENCV
#include "mediapipe/calculators/image/affine_transformation_runner_opencv.h"
#endif  // !MEDIAPIPE_DISABLE_OPENCV
#include "mediapipe/calculators/image/warp_affine_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/ret_check.h"
#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace mediapipe {

namespace {

AffineTransformation::BorderMode GetBorderMode(
    mediapipe::WarpAffineCalculatorOptions::BorderMode border_mode) {
  switch (border_mode) {
    case mediapipe::WarpAffineCalculatorOptions::BORDER_ZERO:
      return AffineTransformation::BorderMode::kZero;
    case mediapipe::WarpAffineCalculatorOptions::BORDER_UNSPECIFIED:
    case mediapipe::WarpAffineCalculatorOptions::BORDER_REPLICATE:
      return AffineTransformation::BorderMode::kReplicate;
  }
}

template <typename ImageT>
class WarpAffineRunnerHolder {};

#if !MEDIAPIPE_DISABLE_OPENCV
template <>
class WarpAffineRunnerHolder<ImageFrame> {
 public:
  using RunnerType = AffineTransformation::Runner<ImageFrame, ImageFrame>;
  absl::Status Open(CalculatorContext* cc) { return absl::OkStatus(); }
  absl::StatusOr<RunnerType*> GetRunner() {
    if (!runner_) {
      ASSIGN_OR_RETURN(runner_, CreateAffineTransformationOpenCvRunner());
    }
    return runner_.get();
  }

 private:
  std::unique_ptr<RunnerType> runner_;
};
#endif  // !MEDIAPIPE_DISABLE_OPENCV

#if !MEDIAPIPE_DISABLE_GPU
template <>
class WarpAffineRunnerHolder<mediapipe::GpuBuffer> {
 public:
  using RunnerType =
      AffineTransformation::Runner<mediapipe::GpuBuffer,
                                   std::unique_ptr<mediapipe::GpuBuffer>>;
  absl::Status Open(CalculatorContext* cc) {
    gpu_origin_ =
        cc->Options<mediapipe::WarpAffineCalculatorOptions>().gpu_origin();
    gl_helper_ = std::make_shared<mediapipe::GlCalculatorHelper>();
    return gl_helper_->Open(cc);
  }
  absl::StatusOr<RunnerType*> GetRunner() {
    if (!runner_) {
      ASSIGN_OR_RETURN(
          runner_, CreateAffineTransformationGlRunner(gl_helper_, gpu_origin_));
    }
    return runner_.get();
  }

 private:
  mediapipe::GpuOrigin::Mode gpu_origin_;
  std::shared_ptr<mediapipe::GlCalculatorHelper> gl_helper_;
  std::unique_ptr<RunnerType> runner_;
};
#endif  // !MEDIAPIPE_DISABLE_GPU

template <>
class WarpAffineRunnerHolder<mediapipe::Image> {
 public:
  absl::Status Open(CalculatorContext* cc) { return runner_.Open(cc); }
  absl::StatusOr<
      AffineTransformation::Runner<mediapipe::Image, mediapipe::Image>*>
  GetRunner() {
    return &runner_;
  }

 private:
  class Runner : public AffineTransformation::Runner<mediapipe::Image,
                                                     mediapipe::Image> {
   public:
    absl::Status Open(CalculatorContext* cc) {
#if !MEDIAPIPE_DISABLE_OPENCV
      MP_RETURN_IF_ERROR(cpu_holder_.Open(cc));
#endif  // !MEDIAPIPE_DISABLE_OPENCV
#if !MEDIAPIPE_DISABLE_GPU
      MP_RETURN_IF_ERROR(gpu_holder_.Open(cc));
#endif  // !MEDIAPIPE_DISABLE_GPU
      return absl::OkStatus();
    }
    absl::StatusOr<mediapipe::Image> Run(
        const mediapipe::Image& input, const std::array<float, 16>& matrix,
        const AffineTransformation::Size& size,
        AffineTransformation::BorderMode border_mode) override {
      if (input.UsesGpu()) {
#if !MEDIAPIPE_DISABLE_GPU
        ASSIGN_OR_RETURN(auto* runner, gpu_holder_.GetRunner());
        ASSIGN_OR_RETURN(auto result, runner->Run(input.GetGpuBuffer(), matrix,
                                                  size, border_mode));
        return mediapipe::Image(*result);
#else
        return absl::UnavailableError("GPU support is disabled");
#endif  // !MEDIAPIPE_DISABLE_GPU
      }
#if !MEDIAPIPE_DISABLE_OPENCV
      ASSIGN_OR_RETURN(auto* runner, cpu_holder_.GetRunner());
      const auto& frame_ptr = input.GetImageFrameSharedPtr();
      // Wrap image into image frame.
      const ImageFrame image_frame(frame_ptr->Format(), frame_ptr->Width(),
                                   frame_ptr->Height(), frame_ptr->WidthStep(),
                                   const_cast<uint8_t*>(frame_ptr->PixelData()),
                                   [](uint8* data){});
      ASSIGN_OR_RETURN(auto result,
                       runner->Run(image_frame, matrix, size, border_mode));
      return mediapipe::Image(std::make_shared<ImageFrame>(std::move(result)));
#else
      return absl::UnavailableError("OpenCV support is disabled");
#endif  // !MEDIAPIPE_DISABLE_OPENCV
    }

   private:
#if !MEDIAPIPE_DISABLE_OPENCV
    WarpAffineRunnerHolder<ImageFrame> cpu_holder_;
#endif  // !MEDIAPIPE_DISABLE_OPENCV
#if !MEDIAPIPE_DISABLE_GPU
    WarpAffineRunnerHolder<mediapipe::GpuBuffer> gpu_holder_;
#endif  // !MEDIAPIPE_DISABLE_GPU
  };

  Runner runner_;
};

template <typename InterfaceT>
class WarpAffineCalculatorImpl : public mediapipe::api2::NodeImpl<InterfaceT> {
 public:
#if !MEDIAPIPE_DISABLE_GPU
  static absl::Status UpdateContract(CalculatorContract* cc) {
    if constexpr (std::is_same_v<InterfaceT, WarpAffineCalculatorGpu> ||
                  std::is_same_v<InterfaceT, WarpAffineCalculator>) {
      MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
    }
    return absl::OkStatus();
  }
#endif  // !MEDIAPIPE_DISABLE_GPU

  absl::Status Open(CalculatorContext* cc) override { return holder_.Open(cc); }

  absl::Status Process(CalculatorContext* cc) override {
    if (InterfaceT::kInImage(cc).IsEmpty() ||
        InterfaceT::kMatrix(cc).IsEmpty() ||
        InterfaceT::kOutputSize(cc).IsEmpty()) {
      return absl::OkStatus();
    }
    const std::array<float, 16>& transform = *InterfaceT::kMatrix(cc);
    auto [out_width, out_height] = *InterfaceT::kOutputSize(cc);
    AffineTransformation::Size output_size;
    output_size.width = out_width;
    output_size.height = out_height;
    ASSIGN_OR_RETURN(auto* runner, holder_.GetRunner());
    ASSIGN_OR_RETURN(
        auto result,
        runner->Run(
            *InterfaceT::kInImage(cc), transform, output_size,
            GetBorderMode(cc->Options<mediapipe::WarpAffineCalculatorOptions>()
                              .border_mode())));
    InterfaceT::kOutImage(cc).Send(std::move(result));

    return absl::OkStatus();
  }

 private:
  WarpAffineRunnerHolder<typename decltype(InterfaceT::kInImage)::PayloadT>
      holder_;
};

}  // namespace

#if !MEDIAPIPE_DISABLE_OPENCV
MEDIAPIPE_NODE_IMPLEMENTATION(
    WarpAffineCalculatorImpl<WarpAffineCalculatorCpu>);
#endif  // !MEDIAPIPE_DISABLE_OPENCV
#if !MEDIAPIPE_DISABLE_GPU
MEDIAPIPE_NODE_IMPLEMENTATION(
    WarpAffineCalculatorImpl<WarpAffineCalculatorGpu>);
#endif  // !MEDIAPIPE_DISABLE_GPU
MEDIAPIPE_NODE_IMPLEMENTATION(WarpAffineCalculatorImpl<WarpAffineCalculator>);

}  // namespace mediapipe

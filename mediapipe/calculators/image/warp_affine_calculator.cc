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
#include <type_traits>
#include <utility>

#include "mediapipe/calculators/image/affine_transformation.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/api3/calculator_contract.h"
#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/calculators/image/affine_transformation_runner_gl.h"
#endif  // !MEDIAPIPE_DISABLE_GPU
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#if !MEDIAPIPE_DISABLE_OPENCV
#include "mediapipe/calculators/image/affine_transformation_runner_opencv.h"
#endif  // !MEDIAPIPE_DISABLE_OPENCV
#include "mediapipe/calculators/image/warp_affine_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/status_macros.h"
#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_service.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace mediapipe::api3 {

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

AffineTransformation::Interpolation GetInterpolation(
    mediapipe::WarpAffineCalculatorOptions::Interpolation interpolation) {
  switch (interpolation) {
    case mediapipe::WarpAffineCalculatorOptions::INTER_UNSPECIFIED:
    case mediapipe::WarpAffineCalculatorOptions::INTER_LINEAR:
      return AffineTransformation::Interpolation::kLinear;
    case mediapipe::WarpAffineCalculatorOptions::INTER_CUBIC:
      return AffineTransformation::Interpolation::kCubic;
  }
}

template <typename ImageT>
class WarpAffineRunnerHolder {};

#if !MEDIAPIPE_DISABLE_OPENCV

template <>
class WarpAffineRunnerHolder<ImageFrame> {
 public:
  using RunnerType = AffineTransformation::Runner<ImageFrame, ImageFrame>;

  absl::Status Open(mediapipe::CalculatorContext& cc,
                    const mediapipe::WarpAffineCalculatorOptions& options) {
    interpolation_ = GetInterpolation(options.interpolation());
    return absl::OkStatus();
  }

  absl::StatusOr<RunnerType*> GetRunner() {
    if (!runner_) {
      MP_ASSIGN_OR_RETURN(
          runner_, CreateAffineTransformationOpenCvRunner(interpolation_));
    }
    return runner_.get();
  }

 private:
  std::unique_ptr<RunnerType> runner_;
  AffineTransformation::Interpolation interpolation_;
};

#endif  // !MEDIAPIPE_DISABLE_OPENCV

#if !MEDIAPIPE_DISABLE_GPU

template <>
class WarpAffineRunnerHolder<mediapipe::GpuBuffer> {
 public:
  using RunnerType =
      AffineTransformation::Runner<mediapipe::GpuBuffer,
                                   std::unique_ptr<mediapipe::GpuBuffer>>;

  absl::Status Open(mediapipe::CalculatorContext& cc,
                    const mediapipe::WarpAffineCalculatorOptions& options) {
    gpu_origin_ = options.gpu_origin();
    gl_helper_ = std::make_shared<mediapipe::GlCalculatorHelper>();
    interpolation_ = GetInterpolation(options.interpolation());
    return gl_helper_->Open(&cc);
  }

  absl::StatusOr<RunnerType*> GetRunner() {
    if (!runner_) {
      MP_ASSIGN_OR_RETURN(
          runner_, CreateAffineTransformationGlRunner(gl_helper_, gpu_origin_,
                                                      interpolation_));
    }
    return runner_.get();
  }

 private:
  mediapipe::GpuOrigin::Mode gpu_origin_;
  std::shared_ptr<mediapipe::GlCalculatorHelper> gl_helper_;
  std::unique_ptr<RunnerType> runner_;
  AffineTransformation::Interpolation interpolation_;
};

#endif  // !MEDIAPIPE_DISABLE_GPU

template <>
class WarpAffineRunnerHolder<mediapipe::Image> {
 public:
  using RunnerType =
      AffineTransformation::Runner<mediapipe::Image, mediapipe::Image>;

  absl::Status Open(mediapipe::CalculatorContext& cc,
                    const mediapipe::WarpAffineCalculatorOptions& options) {
    return runner_.Open(cc, options);
  }

  absl::StatusOr<RunnerType*> GetRunner() { return &runner_; }

 private:
  class Runner : public AffineTransformation::Runner<mediapipe::Image,
                                                     mediapipe::Image> {
   public:
    absl::Status Open(mediapipe::CalculatorContext& cc,
                      const mediapipe::WarpAffineCalculatorOptions& options) {
#if !MEDIAPIPE_DISABLE_OPENCV
      MP_RETURN_IF_ERROR(cpu_holder_.Open(cc, options));
#endif  // !MEDIAPIPE_DISABLE_OPENCV

#if !MEDIAPIPE_DISABLE_GPU
      if (cc.Service(kGpuService).IsAvailable()) {
        MP_RETURN_IF_ERROR(gpu_holder_.Open(cc, options));
        gpu_holder_initialized_ = true;
      }
#endif  // !MEDIAPIPE_DISABLE_GPU

      return absl::OkStatus();
    }

    absl::StatusOr<mediapipe::Image> Run(
        const mediapipe::Image& input, const std::array<float, 16>& matrix,
        const AffineTransformation::Size& size,
        AffineTransformation::BorderMode border_mode) override {
      if (input.UsesGpu()) {
#if !MEDIAPIPE_DISABLE_GPU
        if (!gpu_holder_initialized_) {
          return absl::UnavailableError("GPU support is not available");
        }
        MP_ASSIGN_OR_RETURN(auto* runner, gpu_holder_.GetRunner());
        MP_ASSIGN_OR_RETURN(
            auto result,
            runner->Run(input.GetGpuBuffer(), matrix, size, border_mode));
        return mediapipe::Image(*result);
#else
        return absl::UnavailableError("GPU support is disabled");
#endif  // !MEDIAPIPE_DISABLE_GPU
      }
#if !MEDIAPIPE_DISABLE_OPENCV
      MP_ASSIGN_OR_RETURN(auto* runner, cpu_holder_.GetRunner());
      const auto& frame_ptr = input.GetImageFrameSharedPtr();
      // Wrap image into image frame.
      const ImageFrame image_frame(frame_ptr->Format(), frame_ptr->Width(),
                                   frame_ptr->Height(), frame_ptr->WidthStep(),
                                   const_cast<uint8_t*>(frame_ptr->PixelData()),
                                   [](uint8_t* data){});
      MP_ASSIGN_OR_RETURN(auto result,
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
    bool gpu_holder_initialized_ = false;
#endif  // !MEDIAPIPE_DISABLE_GPU
  };

  Runner runner_;
};

template <typename ImageT>
class WarpAffineNodeImpl
    : public Calculator<WarpAffineNode<ImageT>, WarpAffineNodeImpl<ImageT>> {
 public:
#if !MEDIAPIPE_DISABLE_GPU
  static absl::Status UpdateContract(
      CalculatorContract<WarpAffineNode<ImageT>>& cc) {
    if constexpr (std::is_same_v<ImageT, GpuBuffer> ||
                  std::is_same_v<ImageT, Image>) {
      MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(
          &cc.GetGenericContract(), /*request_gpu_as_optional=*/true));
    }
    return absl::OkStatus();
  }
#endif  // !MEDIAPIPE_DISABLE_GPU

  absl::Status Process(CalculatorContext<WarpAffineNode<ImageT>>& cc) override {
    if (!cc.in_image || !cc.matrix || !cc.output_size) {
      return absl::OkStatus();
    }

    if (!holder_initialized_) {
      MP_RETURN_IF_ERROR(
          holder_.Open(cc.GetGenericContext(), cc.options.Get()));
      holder_initialized_ = true;
    }

    const std::array<float, 16>& transform = cc.matrix.GetOrDie();
    auto [out_width, out_height] = cc.output_size.GetOrDie();
    AffineTransformation::Size output_size;
    output_size.width = out_width;
    output_size.height = out_height;
    MP_ASSIGN_OR_RETURN(auto* runner, holder_.GetRunner());
    MP_ASSIGN_OR_RETURN(
        auto result,
        runner->Run(cc.in_image.GetOrDie(), transform, output_size,
                    GetBorderMode(cc.options.Get().border_mode())));
    cc.out_image.Send(std::move(result));

    return absl::OkStatus();
  }

 private:
  WarpAffineRunnerHolder<ImageT> holder_;
  bool holder_initialized_ = false;
};

#if !MEDIAPIPE_DISABLE_OPENCV
// Explicit instantiation for ImageFrame.
template class WarpAffineNodeImpl<ImageFrame>;
#endif  // !MEDIAPIPE_DISABLE_OPENCV

#if !MEDIAPIPE_DISABLE_GPU
// Explicit instantiation for GpuBuffer.
template class WarpAffineNodeImpl<GpuBuffer>;
#endif  // !MEDIAPIPE_DISABLE_GPU

// Explicit instantiation for Image.
template class WarpAffineNodeImpl<Image>;

}  // namespace
}  // namespace mediapipe::api3

#ifndef MEDIAPIPE_CALCULATORS_IMAGE_IMAGE_CROPPING_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_IMAGE_IMAGE_CROPPING_CALCULATOR_H_

#include "mediapipe/calculators/image/image_cropping_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"

#if !defined(MEDIAPIPE_DISABLE_GPU)
#include "mediapipe/gpu/gl_calculator_helper.h"
#endif  //  !MEDIAPIPE_DISABLE_GPU

// Crops the input texture to the given rectangle region. The rectangle can
// be at arbitrary location on the image with rotation. If there's rotation, the
// output texture will have the size of the input rectangle. The rotation should
// be in radian, see rect.proto for detail.
//
// Input:
//   One of the following two tags:
//   IMAGE - ImageFrame representing the input image.
//   IMAGE_GPU - GpuBuffer representing the input image.
//   One of the following two tags (optional if WIDTH/HEIGHT is specified):
//   RECT - A Rect proto specifying the width/height and location of the
//          cropping rectangle.
//   NORM_RECT - A NormalizedRect proto specifying the width/height and location
//               of the cropping rectangle in normalized coordinates.
//   Alternative tags to RECT (optional if RECT/NORM_RECT is specified):
//   WIDTH - The desired width of the output cropped image,
//           based on image center
//   HEIGHT - The desired height of the output cropped image,
//            based on image center
//
// Output:
//   One of the following two tags:
//   IMAGE - Cropped ImageFrame
//   IMAGE_GPU - Cropped GpuBuffer.
//
// Note: input_stream values take precedence over options defined in the graph.
//
namespace mediapipe {

struct RectSpec {
  int width;
  int height;
  int center_x;
  int center_y;
  float rotation;

  bool operator==(const RectSpec& rect) const {
    return (width == rect.width && height == rect.height &&
            center_x == rect.center_x && center_y == rect.center_y &&
            rotation == rect.rotation);
  }
};

class ImageCroppingCalculator : public CalculatorBase {
 public:
  ImageCroppingCalculator() = default;
  ~ImageCroppingCalculator() override = default;

  static ::mediapipe::Status GetContract(CalculatorContract* cc);
  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  ::mediapipe::Status Close(CalculatorContext* cc) override;
  static RectSpec GetCropSpecs(const CalculatorContext* cc, int src_width,
                               int src_height);

 private:
  ::mediapipe::Status ValidateBorderModeForCPU(CalculatorContext* cc);
  ::mediapipe::Status ValidateBorderModeForGPU(CalculatorContext* cc);
  ::mediapipe::Status RenderCpu(CalculatorContext* cc);
  ::mediapipe::Status RenderGpu(CalculatorContext* cc);
  ::mediapipe::Status InitGpu(CalculatorContext* cc);
  void GlRender();
  void GetOutputDimensions(CalculatorContext* cc, int src_width, int src_height,
                           int* dst_width, int* dst_height);
  ::mediapipe::Status GetBorderModeForOpenCV(CalculatorContext* cc,
                                             int* border_mode);

  mediapipe::ImageCroppingCalculatorOptions options_;

  bool use_gpu_ = false;
  // Output texture corners (4) after transoformation in normalized coordinates.
  float transformed_points_[8];
#if !defined(MEDIAPIPE_DISABLE_GPU)
  bool gpu_initialized_ = false;
  mediapipe::GlCalculatorHelper gpu_helper_;
  GLuint program_ = 0;
#endif  //  !MEDIAPIPE_DISABLE_GPU
};

}  // namespace mediapipe
#endif  // MEDIAPIPE_CALCULATORS_IMAGE_IMAGE_CROPPING_CALCULATOR_H_

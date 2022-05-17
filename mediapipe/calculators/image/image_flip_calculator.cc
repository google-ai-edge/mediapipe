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

//#include "mediapipe/calculators/image/image_clone_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/status.h"

// akash

#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
//#include "mediapipe/framework/formats/video_stream_header.h"
// akash

#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_calculator_helper.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace mediapipe {
namespace api2 {

#if MEDIAPIPE_DISABLE_GPU
// Just a placeholder to not have to depend on mediapipe::GpuBuffer.
using GpuBuffer = AnyType;
#else
using GpuBuffer = mediapipe::GpuBuffer;
#endif  // MEDIAPIPE_DISABLE_GPU

// Clones an input image and makes sure in the output clone the pixel data are
// stored on the target storage (CPU vs GPU) specified in the calculator option.
//
// The clone shares ownership of the input pixel data on the existing storage.
// If the target storage is different from the existing one, then the data is
// further copied there.
//
// Example usage:
// node {
//   calculator: "ImageCloneCalculator"
//   input_stream: "input"
//   output_stream: "output"
//   options: {
//     [mediapipe.ImageCloneCalculatorOptions.ext] {
//       output_on_gpu: true
//     }
//   }
// }
class CustomImageFlipCalculator : public Node {
 public:
  static constexpr Input<ImageFrame> kIn{"IMAGE"};
  static constexpr Output<ImageFrame> kOut{"IMAGE"};
  
  MEDIAPIPE_NODE_CONTRACT(kIn, kOut);

  static absl::Status UpdateContract(CalculatorContract* cc) {
#if MEDIAPIPE_DISABLE_GPU
    /*if (cc->Options<mediapipe::ImageCloneCalculatorOptions>().output_on_gpu()) {
      return absl::UnimplementedError(
          "GPU processing is disabled in build flags");
    }*/
        
#else
    MP_RETURN_IF_ERROR(mediapipe::GlCalculatorHelper::UpdateContract(cc));
#endif  // MEDIAPIPE_DISABLE_GPU
    return absl::OkStatus();
  }

    absl::Status Open(CalculatorContext* cc) override {
    /*
    const auto& options = cc->Options<mediapipe::ImageCloneCalculatorOptions>();
    output_on_gpu_ = options.output_on_gpu();
#if !MEDIAPIPE_DISABLE_GPU
    MP_RETURN_IF_ERROR(gpu_helper_.Open(cc));
#endif  // !MEDIAPIPE_DISABLE_GPU*/
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
      std::cout<<"\n inside image_flip_calculator  process";
      float scale_fact=1;
      int output_width=0;
      int output_height=0;
      mediapipe::ImageFormat::Format format;
      cv::Mat input_mat;
      cv::Mat scaled_mat;
      cv::Mat flipped_mat;

    std::unique_ptr<ImageFrame> output;
    const auto& input = *kIn(cc);
    if (FALSE) {//
#if !MEDIAPIPE_DISABLE_GPU
      // Create an output Image that co-owns the underlying texture buffer as
      // the input Image.
      output = std::make_unique<Image>(input.GetGpuBuffer());
#endif  // !MEDIAPIPE_DISABLE_GPU
    } else {
        
      // Make a copy of the input packet to co-own the input Image.
      mediapipe::Packet* packet_copy_ptr =
          new mediapipe::Packet(kIn(cc).packet());
      // Create an output Image that (co-)owns a new ImageFrame that points to
      // the same pixel data as the input Image and also owns the packet
      // copy. As a result, the output Image indirectly co-owns the input
      // Image. This ensures a correct life span of the shared pixel data.
        
        // copde to flip
        

        output_width=input.Width();
        output_height=input.Height();
        input_mat = formats::MatView(&input);
        
        format = input.Format();
        /*cv::Rect myROI(0, 0, output_width/2, output_height/2);
        cv::Mat croppedRef(input_mat, myROI);
        cv::Mat cropped;
        croppedRef.copyTo(cropped);
        cv::Mat imgPanel(480, 640, CV_8UC1, Scalar(0));
        Mat imgPanelRoi(imgPanel, Rect(0, 0, imgSrc.cols, imgSrc.rows));
        imgSrc.copyTo(imgPanelRoi);
        cv::resize(cropped, input_mat, cv::Size(), 2, 2);
        std::cout<<"size>>>>"<<input_mat.size;*/
        



            
        // code to flip
        cv::flip(input_mat, flipped_mat, 1);
// Use Flip code 0 to flip vertically

        output = absl::make_unique<mediapipe::ImageFrame>(
         mediapipe::ImageFormat::SRGB,
         int(output_width),
         int(output_height),
         mediapipe::ImageFrame::kGlDefaultAlignmentBoundary
       );
        flipped_mat.copyTo(formats::MatView(output.get()));
    }

    if (output_on_gpu_) {
#if !MEDIAPIPE_DISABLE_GPU
      gpu_helper_.RunInGlContext([&output]() { output->ConvertToGpu(); });
#endif  // !MEDIAPIPE_DISABLE_GPU
    } else {
      //output->ConvertToCpu();
        
    }
      //ASSIGN_OR_RETURN(output,);

      kOut(cc).Send(std::move(output));

    return absl::OkStatus();
  }

 private:
  bool output_on_gpu_;
#if !MEDIAPIPE_DISABLE_GPU
  mediapipe::GlCalculatorHelper gpu_helper_;
#endif  // !MEDIAPIPE_DISABLE_GPU
};
MEDIAPIPE_REGISTER_NODE(CustomImageFlipCalculator);

}  // namespace api2
}  // namespace mediapipe

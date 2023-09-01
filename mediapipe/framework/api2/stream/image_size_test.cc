#include "mediapipe/framework/api2/stream/image_size.h"

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/gpu/gpu_buffer.h"

namespace mediapipe::api2::builder {
namespace {

TEST(GetImageSize, VerifyConfig) {
  Graph graph;

  Stream<ImageFrame> image_frame = graph.In("IMAGE_FRAME").Cast<ImageFrame>();
  image_frame.SetName("image_frame");
  Stream<GpuBuffer> gpu_buffer = graph.In("GPU_BUFFER").Cast<GpuBuffer>();
  gpu_buffer.SetName("gpu_buffer");
  Stream<mediapipe::Image> image = graph.In("IMAGE").Cast<mediapipe::Image>();
  image.SetName("image");

  GetImageSize(image_frame, graph).SetName("image_frame_size");
  GetImageSize(gpu_buffer, graph).SetName("gpu_buffer_size");
  GetImageSize(image, graph).SetName("image_size");

  EXPECT_THAT(
      graph.GetConfig(),
      EqualsProto(mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        node {
          calculator: "ImagePropertiesCalculator"
          input_stream: "IMAGE:image_frame"
          output_stream: "SIZE:image_frame_size"
        }
        node {
          calculator: "ImagePropertiesCalculator"
          input_stream: "IMAGE_GPU:gpu_buffer"
          output_stream: "SIZE:gpu_buffer_size"
        }
        node {
          calculator: "ImagePropertiesCalculator"
          input_stream: "IMAGE:image"
          output_stream: "SIZE:image_size"
        }
        input_stream: "GPU_BUFFER:gpu_buffer"
        input_stream: "IMAGE:image"
        input_stream: "IMAGE_FRAME:image_frame"
      )pb")));

  CalculatorGraph calcualtor_graph;
  MP_EXPECT_OK(calcualtor_graph.Initialize(graph.GetConfig()));
}
}  // namespace
}  // namespace mediapipe::api2::builder

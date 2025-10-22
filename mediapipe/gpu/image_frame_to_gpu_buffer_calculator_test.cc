#include "mediapipe/gpu/image_frame_to_gpu_buffer_calculator.h"

#include <utility>

#include "mediapipe/framework/api3/function_runner.h"
#include "mediapipe/framework/api3/graph.h"
#include "mediapipe/framework/api3/packet.h"
#include "mediapipe/framework/api3/stream.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/test_util.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_service.h"
#include "mediapipe/gpu/gpu_test_base.h"
#include "mediapipe/util/image_test_utils.h"

namespace mediapipe {
namespace {

using ::mediapipe::api3::GenericGraph;
using ::mediapipe::api3::ImageFrameToGpuBufferNode;
using ::mediapipe::api3::Packet;
using ::mediapipe::api3::Runner;
using ::mediapipe::api3::Stream;

class ImageFrameToGpuBufferCalculatorTest : public GpuTestBase {};

TEST_F(ImageFrameToGpuBufferCalculatorTest, ConvertsImageFrame) {
  auto graph_builder_fn = [](GenericGraph& graph,
                             Stream<ImageFrame> in) -> Stream<GpuBuffer> {
    auto& node = graph.AddNode<ImageFrameToGpuBufferNode>();
    node.image_frame.Set(in);
    return node.gpu_buffer.Get();
  };
  MP_ASSERT_OK_AND_ASSIGN(auto runner,
                          Runner::For(std::move(graph_builder_fn))
                              .SetService(kGpuService, gpu_resources_)
                              .Create());

  constexpr int kWidth = 8;
  constexpr int kHeight = 8;
  ImageFrame input_image(ImageFormat::SRGBA, kWidth, kHeight);
  FillImageFrameRGBA(input_image, 255, 0, 0, 255);

  MP_ASSERT_OK_AND_ASSIGN(
      Packet<GpuBuffer> out_packet,
      runner.Run(api3::MakePacket<ImageFrame>(std::move(input_image))));
  ASSERT_TRUE(out_packet);
  const auto& gpu_buffer = out_packet.GetOrDie();

  ASSERT_EQ(gpu_buffer.width(), kWidth);
  ASSERT_EQ(gpu_buffer.height(), kHeight);

  ImageFrame red(ImageFormat::SRGBA, kWidth, kHeight);
  FillImageFrameRGBA(red, 255, 0, 0, 255);

  auto view = gpu_buffer.GetReadView<ImageFrame>();
  mediapipe::ImageFrameComparisonOptions opts = {
      .max_color_diff = 0.0f,
      .max_avg_diff = 0.0f,
  };
  MP_EXPECT_OK(CompareAndSaveImageOutputDynamic(*view, red, opts));
}

}  // namespace
}  // namespace mediapipe

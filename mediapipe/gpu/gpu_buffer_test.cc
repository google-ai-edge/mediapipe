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

#include "mediapipe/gpu/gpu_buffer.h"

#include <utility>

#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/tool/test_util.h"
#include "mediapipe/gpu/gl_texture_buffer.h"
#include "mediapipe/gpu/gl_texture_util.h"
#include "mediapipe/gpu/gpu_buffer_storage_ahwb.h"
#include "mediapipe/gpu/gpu_buffer_storage_image_frame.h"
#include "mediapipe/gpu/gpu_test_base.h"
#include "stb_image.h"
#include "stb_image_write.h"

namespace mediapipe {
namespace {

void FillImageFrameRGBA(ImageFrame& image, uint8_t r, uint8_t g, uint8_t b,
                        uint8_t a) {
  auto* data = image.MutablePixelData();
  for (int y = 0; y < image.Height(); ++y) {
    auto* row = data + image.WidthStep() * y;
    for (int x = 0; x < image.Width(); ++x) {
      auto* pixel = row + x * image.NumberOfChannels();
      pixel[0] = r;
      pixel[1] = g;
      pixel[2] = b;
      pixel[3] = a;
    }
  }
}

class GpuBufferTest : public GpuTestBase {};

TEST_F(GpuBufferTest, BasicTest) {
  RunInGlContext([this] {
    MP_ASSERT_OK_AND_ASSIGN(GpuBuffer buffer,
                            gpu_shared_.gpu_buffer_pool.GetBuffer(300, 200));
    EXPECT_EQ(buffer.width(), 300);
    EXPECT_EQ(buffer.height(), 200);
    EXPECT_TRUE(buffer);
    EXPECT_FALSE(buffer == nullptr);

    GpuBuffer no_buffer;
    EXPECT_FALSE(no_buffer);
    EXPECT_TRUE(no_buffer == nullptr);

    GpuBuffer buffer2 = buffer;
    EXPECT_EQ(buffer, buffer);
    EXPECT_EQ(buffer, buffer2);
    EXPECT_NE(buffer, no_buffer);

    buffer = nullptr;
    EXPECT_TRUE(buffer == nullptr);
    EXPECT_TRUE(buffer == no_buffer);
  });
}

TEST_F(GpuBufferTest, GlTextureView) {
  GpuBuffer buffer(300, 200, GpuBufferFormat::kBGRA32);
  EXPECT_EQ(buffer.width(), 300);
  EXPECT_EQ(buffer.height(), 200);
  EXPECT_TRUE(buffer);
  EXPECT_FALSE(buffer == nullptr);

  RunInGlContext([&buffer] {
    TempGlFramebuffer fb;
    auto view = buffer.GetWriteView<GlTextureView>(0);
    FillGlTextureRgba(view, 1.0, 0.0, 0.0, 1.0);
    glFlush();
  });
  std::shared_ptr<const ImageFrame> view = buffer.GetReadView<ImageFrame>();
  EXPECT_EQ(view->Width(), 300);
  EXPECT_EQ(view->Height(), 200);

  ImageFrame red(ImageFormat::SRGBA, 300, 200);
  FillImageFrameRGBA(red, 255, 0, 0, 255);

  EXPECT_TRUE(CompareImageFrames(*view, red, 0.0, 0.0));
  MP_EXPECT_OK(SavePngTestOutput(red, "gltv_red_gold"));
  MP_EXPECT_OK(SavePngTestOutput(*view, "gltv_red_view"));
}

TEST_F(GpuBufferTest, ImageFrame) {
  GpuBuffer buffer(300, 200, GpuBufferFormat::kBGRA32);
  EXPECT_EQ(buffer.width(), 300);
  EXPECT_EQ(buffer.height(), 200);
  EXPECT_TRUE(buffer);
  EXPECT_FALSE(buffer == nullptr);

  {
    std::shared_ptr<ImageFrame> view = buffer.GetWriteView<ImageFrame>();
    EXPECT_EQ(view->Width(), 300);
    EXPECT_EQ(view->Height(), 200);
    FillImageFrameRGBA(*view, 255, 0, 0, 255);
  }

  GpuBuffer buffer2(300, 200, GpuBufferFormat::kBGRA32);
  RunInGlContext([&buffer, &buffer2] {
    TempGlFramebuffer fb;
    auto src = buffer.GetReadView<GlTextureView>(0);
    auto dst = buffer2.GetWriteView<GlTextureView>(0);
    CopyGlTexture(src, dst);
    glFlush();
  });
  {
    std::shared_ptr<const ImageFrame> view = buffer2.GetReadView<ImageFrame>();
    EXPECT_EQ(view->Width(), 300);
    EXPECT_EQ(view->Height(), 200);

    ImageFrame red(ImageFormat::SRGBA, 300, 200);
    FillImageFrameRGBA(red, 255, 0, 0, 255);

    EXPECT_TRUE(CompareImageFrames(*view, red, 0.0, 0.0));
    MP_EXPECT_OK(SavePngTestOutput(red, "if_red_gold"));
    MP_EXPECT_OK(SavePngTestOutput(*view, "if_red_view"));
  }
}

TEST_F(GpuBufferTest, Overwrite) {
  GpuBuffer buffer(300, 200, GpuBufferFormat::kBGRA32);
  EXPECT_EQ(buffer.width(), 300);
  EXPECT_EQ(buffer.height(), 200);
  EXPECT_TRUE(buffer);
  EXPECT_FALSE(buffer == nullptr);

  {
    std::shared_ptr<ImageFrame> view = buffer.GetWriteView<ImageFrame>();
    EXPECT_EQ(view->Width(), 300);
    EXPECT_EQ(view->Height(), 200);
    FillImageFrameRGBA(*view, 255, 0, 0, 255);
  }

  GpuBuffer red_copy(300, 200, GpuBufferFormat::kBGRA32);
  RunInGlContext([&buffer, &red_copy] {
    TempGlFramebuffer fb;
    auto src = buffer.GetReadView<GlTextureView>(0);
    auto dst = red_copy.GetWriteView<GlTextureView>(0);
    CopyGlTexture(src, dst);
    glFlush();
  });

  {
    std::shared_ptr<const ImageFrame> view = red_copy.GetReadView<ImageFrame>();
    ImageFrame red(ImageFormat::SRGBA, 300, 200);
    FillImageFrameRGBA(red, 255, 0, 0, 255);

    EXPECT_TRUE(CompareImageFrames(*view, red, 0.0, 0.0));
    MP_EXPECT_OK(SavePngTestOutput(red, "ow_red_gold"));
    MP_EXPECT_OK(SavePngTestOutput(*view, "ow_red_view"));
  }

  {
    std::shared_ptr<ImageFrame> view = buffer.GetWriteView<ImageFrame>();
    EXPECT_EQ(view->Width(), 300);
    EXPECT_EQ(view->Height(), 200);
    FillImageFrameRGBA(*view, 0, 255, 0, 255);
  }

  GpuBuffer green_copy(300, 200, GpuBufferFormat::kBGRA32);
  RunInGlContext([&buffer, &green_copy] {
    TempGlFramebuffer fb;
    auto src = buffer.GetReadView<GlTextureView>(0);
    auto dst = green_copy.GetWriteView<GlTextureView>(0);
    CopyGlTexture(src, dst);
    glFlush();
  });

  RunInGlContext([&buffer] {
    TempGlFramebuffer fb;
    auto view = buffer.GetWriteView<GlTextureView>(0);
    FillGlTextureRgba(view, 0.0, 0.0, 1.0, 1.0);
    glFlush();
  });

  {
    std::shared_ptr<const ImageFrame> view =
        green_copy.GetReadView<ImageFrame>();
    ImageFrame green(ImageFormat::SRGBA, 300, 200);
    FillImageFrameRGBA(green, 0, 255, 0, 255);

    EXPECT_TRUE(CompareImageFrames(*view, green, 0.0, 0.0));
    MP_EXPECT_OK(SavePngTestOutput(green, "ow_green_gold"));
    MP_EXPECT_OK(SavePngTestOutput(*view, "ow_green_view"));
  }

  {
    std::shared_ptr<const ImageFrame> view = buffer.GetReadView<ImageFrame>();
    ImageFrame blue(ImageFormat::SRGBA, 300, 200);
    FillImageFrameRGBA(blue, 0, 0, 255, 255);

    EXPECT_TRUE(CompareImageFrames(*view, blue, 0.0, 0.0));
    MP_EXPECT_OK(SavePngTestOutput(blue, "ow_blue_gold"));
    MP_EXPECT_OK(SavePngTestOutput(*view, "ow_blue_view"));
  }
}

TEST_F(GpuBufferTest, GlTextureViewRetainsWhatItNeeds) {
  GpuBuffer buffer(300, 200, GpuBufferFormat::kBGRA32);
  {
    std::shared_ptr<ImageFrame> view = buffer.GetWriteView<ImageFrame>();
    EXPECT_EQ(view->Width(), 300);
    EXPECT_EQ(view->Height(), 200);
    FillImageFrameRGBA(*view, 255, 0, 0, 255);
  }

  RunInGlContext([buffer = std::move(buffer)]() mutable {
    // This is not a recommended pattern, but let's make sure that we don't
    // crash if the buffer is released before the view. The view can hold
    // callbacks into its underlying storage.
    auto view = buffer.GetReadView<GlTextureView>(0);
    buffer = nullptr;
  });
  // We're really checking that we haven't crashed.
  EXPECT_TRUE(true);
}

TEST_F(GpuBufferTest, CopiesShareConversions) {
  GpuBuffer buffer(300, 200, GpuBufferFormat::kBGRA32);
  {
    std::shared_ptr<ImageFrame> view = buffer.GetWriteView<ImageFrame>();
    FillImageFrameRGBA(*view, 255, 0, 0, 255);
  }

  GpuBuffer other_handle = buffer;
  RunInGlContext([&buffer] {
    TempGlFramebuffer fb;
    auto view = buffer.GetReadView<GlTextureView>(0);
  });

  // Check that other_handle also sees the same GlTextureBuffer as buffer.
  // Note that this is deliberately written so that it still passes on platforms
  // where we use another storage for GL textures (they will both be null).
  // TODO: expose more accessors for testing?
  EXPECT_EQ(other_handle.internal_storage<GlTextureBuffer>(),
            buffer.internal_storage<GlTextureBuffer>());
}

}  // anonymous namespace
}  // namespace mediapipe

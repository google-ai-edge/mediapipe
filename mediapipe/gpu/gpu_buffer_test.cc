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

#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/tool/test_util.h"
#include "mediapipe/gpu/gpu_buffer_storage_image_frame.h"
#include "mediapipe/gpu/gpu_test_base.h"
#include "stb_image.h"
#include "stb_image_write.h"

namespace mediapipe {
namespace {

void FillImageFrameRGBA(ImageFrame& image, uint8 r, uint8 g, uint8 b, uint8 a) {
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

// Assumes a framebuffer is already set up
void CopyGlTexture(const GlTextureView& src, GlTextureView& dst) {
  glViewport(0, 0, src.width(), src.height());
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, src.target(),
                         src.name(), 0);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(dst.target(), dst.name());
  glCopyTexSubImage2D(dst.target(), 0, 0, 0, 0, 0, dst.width(), dst.height());

  glBindTexture(dst.target(), 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, src.target(), 0,
                         0);
}

void FillGlTextureRgba(GlTextureView& view, float r, float g, float b,
                       float a) {
  glViewport(0, 0, view.width(), view.height());
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, view.target(),
                         view.name(), 0);
  glClearColor(r, g, b, a);
  glClear(GL_COLOR_BUFFER_BIT);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, view.target(), 0,
                         0);
}

class TempGlFramebuffer {
 public:
  TempGlFramebuffer() {
    glGenFramebuffers(1, &framebuffer_);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
  }
  ~TempGlFramebuffer() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &framebuffer_);
  }

 private:
  GLuint framebuffer_;
};

class GpuBufferTest : public GpuTestBase {};

TEST_F(GpuBufferTest, BasicTest) {
  RunInGlContext([this] {
    GpuBuffer buffer = gpu_shared_.gpu_buffer_pool.GetBuffer(300, 200);
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

  EXPECT_TRUE(mediapipe::CompareImageFrames(*view, red, 0.0, 0.0));
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

    EXPECT_TRUE(mediapipe::CompareImageFrames(*view, red, 0.0, 0.0));
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

    EXPECT_TRUE(mediapipe::CompareImageFrames(*view, red, 0.0, 0.0));
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

    EXPECT_TRUE(mediapipe::CompareImageFrames(*view, green, 0.0, 0.0));
    MP_EXPECT_OK(SavePngTestOutput(green, "ow_green_gold"));
    MP_EXPECT_OK(SavePngTestOutput(*view, "ow_green_view"));
  }

  {
    std::shared_ptr<const ImageFrame> view = buffer.GetReadView<ImageFrame>();
    ImageFrame blue(ImageFormat::SRGBA, 300, 200);
    FillImageFrameRGBA(blue, 0, 0, 255, 255);

    EXPECT_TRUE(mediapipe::CompareImageFrames(*view, blue, 0.0, 0.0));
    MP_EXPECT_OK(SavePngTestOutput(blue, "ow_blue_gold"));
    MP_EXPECT_OK(SavePngTestOutput(*view, "ow_blue_view"));
  }
}

}  // anonymous namespace
}  // namespace mediapipe

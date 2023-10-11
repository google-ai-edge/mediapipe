// Copyright 2023 The MediaPipe Authors.
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

#include "Halide.h"
#include "mediapipe/util/frame_buffer/halide/common.h"

namespace {

using ::Halide::BoundaryConditions::repeat_edge;
using ::mediapipe::frame_buffer::halide::common::resize_bilinear_int;

class RgbResize : public Halide::Generator<RgbResize> {
 public:
  Var x{"x"}, y{"y"};

  Input<Buffer<uint8_t, 3>> src_rgb{"src_rgb"};
  Input<float> scale_x{"scale_x", 1.0f, 0.0f, 1024.0f};
  Input<float> scale_y{"scale_y", 1.0f, 0.0f, 1024.0f};

  Output<Func> dst_rgb{"dst_rgb", UInt(8), 3};

  void generate();
  void schedule();
};

void RgbResize::generate() {
  // Resize each of the RGB planes independently.
  resize_bilinear_int(repeat_edge(src_rgb), dst_rgb, scale_x, scale_y);
}

void RgbResize::schedule() {
  Halide::Func dst_rgb_func = dst_rgb;
  Halide::Var c = dst_rgb_func.args()[2];
  Halide::OutputImageParam rgb_output = dst_rgb_func.output_buffer();
  Halide::Expr input_rgb_channels = src_rgb.dim(2).extent();
  Halide::Expr output_rgb_channels = rgb_output.dim(2).extent();
  Halide::Expr min_width =
      Halide::min(src_rgb.dim(0).extent(), rgb_output.dim(0).extent());

  // Specialize the generated code for RGB and RGBA (input and output channels
  // must match); further, specialize the vectorized implementation so it only
  // runs on images wide enough to support it.
  const int vector_size = natural_vector_size<uint8_t>();
  const Expr channel_specializations[] = {
      input_rgb_channels == 3 && output_rgb_channels == 3,
      input_rgb_channels == 4 && output_rgb_channels == 4,
  };
  dst_rgb_func.reorder(c, x, y);
  for (const Expr& channel_specialization : channel_specializations) {
    dst_rgb_func.specialize(channel_specialization && min_width >= vector_size)
        .unroll(c)
        .vectorize(x, vector_size);
  }

  // Require that the input/output buffer be interleaved and tightly-
  // packed; that is, either RGBRGBRGB[...] or RGBARGBARGBA[...],
  // without gaps between pixels.
  src_rgb.dim(0).set_stride(input_rgb_channels);
  src_rgb.dim(2).set_stride(1);
  rgb_output.dim(0).set_stride(output_rgb_channels);
  rgb_output.dim(2).set_stride(1);

  // RGB planes starts at index zero in every dimension.
  src_rgb.dim(0).set_min(0);
  src_rgb.dim(1).set_min(0);
  src_rgb.dim(2).set_min(0);
  rgb_output.dim(0).set_min(0);
  rgb_output.dim(1).set_min(0);
  rgb_output.dim(2).set_min(0);
}

}  // namespace

HALIDE_REGISTER_GENERATOR(RgbResize, rgb_resize_generator)

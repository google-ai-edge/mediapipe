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
using ::mediapipe::frame_buffer::halide::common::is_interleaved;
using ::mediapipe::frame_buffer::halide::common::is_planar;
using ::mediapipe::frame_buffer::halide::common::resize_bilinear_int;

class YuvResize : public Halide::Generator<YuvResize> {
 public:
  Var x{"x"}, y{"y"};

  Input<Buffer<uint8_t, 2>> src_y{"src_y"};
  Input<Buffer<uint8_t, 3>> src_uv{"src_uv"};
  Input<float> scale_x{"scale_x", 1.0f, 0.0f, 1024.0f};
  Input<float> scale_y{"scale_y", 1.0f, 0.0f, 1024.0f};

  Output<Func> dst_y{"dst_y", UInt(8), 2};
  Output<Func> dst_uv{"dst_uv", UInt(8), 3};

  void generate();
  void schedule();
};

void YuvResize::generate() {
  // Resize each of the YUV planes independently.
  resize_bilinear_int(repeat_edge(src_y), dst_y, scale_x, scale_y);
  resize_bilinear_int(repeat_edge(src_uv), dst_uv, scale_x, scale_y);
}

void YuvResize::schedule() {
  // Y plane dimensions start at zero. We could additionally constrain the
  // extent to be even, but that doesn't seem to have any benefit.
  Halide::Func dst_y_func = dst_y;
  Halide::OutputImageParam dst_y_output = dst_y_func.output_buffer();
  src_y.dim(0).set_min(0);
  src_y.dim(1).set_min(0);
  dst_y_output.dim(0).set_min(0);
  dst_y_output.dim(1).set_min(0);

  // UV plane has two channels and is half the size of the Y plane in X/Y.
  Halide::Func dst_uv_func = dst_uv;
  Halide::OutputImageParam dst_uv_output = dst_uv_func.output_buffer();
  src_uv.dim(0).set_bounds(0, (src_y.dim(0).extent() + 1) / 2);
  src_uv.dim(1).set_bounds(0, (src_y.dim(1).extent() + 1) / 2);
  src_uv.dim(2).set_bounds(0, 2);
  dst_uv_output.dim(0).set_bounds(0, (dst_y_output.dim(0).extent() + 1) / 2);
  dst_uv_output.dim(1).set_bounds(0, (dst_y_output.dim(1).extent() + 1) / 2);
  dst_uv_output.dim(2).set_bounds(0, 2);

  // With bilinear filtering enabled, Y plane resize is profitably vectorizable
  // though we must ensure that the image is wide enough to support vector
  // operations.
  const int vector_size = natural_vector_size<uint8_t>();
  Halide::Expr min_y_width =
      Halide::min(src_y.dim(0).extent(), dst_y_output.dim(0).extent());
  dst_y_func.specialize(min_y_width >= vector_size).vectorize(x, vector_size);

  // Remove default memory layout constraints and generate specialized
  // fast-path implementations when both UV source and output are either
  // planar or interleaved. Everything else falls onto a slow path.
  src_uv.dim(0).set_stride(Expr());
  dst_uv_output.dim(0).set_stride(Expr());

  Halide::Var c = dst_uv_func.args()[2];
  dst_uv_func
      .specialize(is_interleaved(src_uv) && is_interleaved(dst_uv_output))
      .reorder(c, x, y)
      .unroll(c);
  dst_uv_func.specialize(is_planar(src_uv) && is_planar(dst_uv_output));
}

}  // namespace

HALIDE_REGISTER_GENERATOR(YuvResize, yuv_resize_generator)

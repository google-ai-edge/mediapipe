#include "mediapipe/util/tflite/operations/resampler.h"

#include <cmath>
#include <cstring>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mediapipe {
namespace tflite_operations {
namespace {

constexpr int kInputTensorSourceIndex = 0;
constexpr int kInputTensorWarpIndex = 1;
constexpr int kOutputTensorDestinationIndex = 0;

// Ported from the tfa implementation in
// tensorflow/tensorflow_addons/custom_ops/image/cc/kernels/resampler_ops.cc
bool ResampleTensors(const float* src, int width, int height, int depth,
                     const float* warp, float* dst) {
  auto get_data_point = [&](const int x, const int y, const int chan) {
    const bool point_is_in_range =
        (x >= 0 && y >= 0 && x <= width - 1 && y <= height - 1);
    return point_is_in_range ? src[(y * width + x) * depth + chan] : 0.0f;
  };
  auto get_warp_point = [&](const int x, const int y, const int chan) {
    const bool point_is_in_range =
        (x >= 0 && y >= 0 && x <= width - 1 && y <= height - 1);
    return point_is_in_range ? warp[(y * width + x) * 2 + chan] : 0.0f;
  };

  for (int dst_y = 0; dst_y < height; ++dst_y) {
    for (int dst_x = 0; dst_x < width; ++dst_x) {
      const float x = get_warp_point(dst_x, dst_y, 0);
      const float y = get_warp_point(dst_x, dst_y, 1);
      float* dst_data = &(dst[(dst_y * width + dst_x) * depth]);

      if (x > static_cast<float>(-1.0) && y > static_cast<float>(-1.0) &&
          x < static_cast<float>(width) && y < static_cast<float>(height)) {
        // Precompute floor (f) and ceil (c) values for x and y.
        const int fx = std::floor(static_cast<float>(x));
        const int fy = std::floor(static_cast<float>(y));
        const int cx = fx + 1;
        const int cy = fy + 1;
        const float dx = static_cast<float>(cx) - x;
        const float dy = static_cast<float>(cy) - y;

        for (int chan = 0; chan < depth; ++chan) {
          const float img_fxfy = dx * dy * get_data_point(fx, fy, chan);
          const float img_cxcy =
              (1.0f - dx) * (1.0f - dy) * get_data_point(cx, cy, chan);
          const float img_fxcy =
              dx * (1.0f - dy) * get_data_point(fx, cy, chan);
          const float img_cxfy =
              (1.0f - dx) * dy * get_data_point(cx, fy, chan);
          dst_data[chan] = img_fxfy + img_cxcy + img_fxcy + img_cxfy;
        }
      } else {
        for (int chan = 0; chan < depth; ++chan) {
          dst_data[chan] = 0.0f;
        }
      }
    }
  }

  return true;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, ::tflite::NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, ::tflite::NumOutputs(node), 1);
  TfLiteTensor* output =
      ::tflite::GetOutput(context, node, kOutputTensorDestinationIndex);
  TF_LITE_ENSURE(context, output != nullptr);
  const TfLiteTensor* source =
      ::tflite::GetInput(context, node, kInputTensorSourceIndex);
  TF_LITE_ENSURE(context, source != nullptr);
  TF_LITE_ENSURE_EQ(context, ::tflite::NumDimensions(source), 4);
  TF_LITE_ENSURE_EQ(context, source->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);

  const TfLiteTensor* warp =
      ::tflite::GetInput(context, node, kInputTensorWarpIndex);
  TF_LITE_ENSURE(context, warp != nullptr);
  TF_LITE_ENSURE_EQ(context, ::tflite::NumDimensions(warp), 4);
  TF_LITE_ENSURE_EQ(context, warp->type, kTfLiteFloat32);

  int batches = source->dims->data[0];
  int height = source->dims->data[1];
  int width = source->dims->data[2];
  int channels_out = source->dims->data[3];

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = batches;
  output_size->data[1] = height;
  output_size->data[2] = width;
  output_size->data[3] = channels_out;

  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* src =
      ::tflite::GetInput(context, node, kInputTensorSourceIndex);
  const TfLiteTensor* warp =
      ::tflite::GetInput(context, node, kInputTensorWarpIndex);
  TfLiteTensor* dst =
      ::tflite::GetOutput(context, node, kOutputTensorDestinationIndex);

  TF_LITE_ENSURE(context, src != nullptr);
  TF_LITE_ENSURE(context, warp != nullptr);
  TF_LITE_ENSURE(context, dst != nullptr);

  // Assumes NHWC layout.
  const int b = src->dims->data[0];
  const int h = src->dims->data[1];
  const int w = src->dims->data[2];
  const int d = src->dims->data[3];

  for (size_t batch = 0; batch < b; ++batch) {
    const size_t warp_offset = h * w * 2 * batch;
    const float* warp_data = ::tflite::GetTensorData<float>(warp) + warp_offset;
    const size_t data_offset = h * w * d * batch;
    const float* src_data = ::tflite::GetTensorData<float>(src) + data_offset;
    float* dst_data = ::tflite::GetTensorData<float>(dst) + data_offset;
    ResampleTensors(src_data, w, h, d, warp_data, dst_data);
  }

  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration* RegisterResampler() {
  static TfLiteRegistration reg = {
      /*.init=*/nullptr,
      /*.free=*/nullptr,
      /*.prepare=*/Prepare,
      /*.invoke=*/Eval,
      /*.profiling_string=*/nullptr,
      /*.builtin_code=*/tflite::BuiltinOperator_CUSTOM,
      /*.custom_name=*/"Resampler",
      /*.version=*/1,
  };
  return &reg;
}

}  // namespace tflite_operations
}  // namespace mediapipe

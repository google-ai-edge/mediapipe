// Copyright 2025 The MediaPipe Authors.
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

#include "mediapipe/framework/debug/logging.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "HalideBuffer.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/absl_log.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

namespace mediapipe::debug {
namespace {

// Returns true if the terminal supports true color.
bool IsTrueColorTerm() {
  const char* colorterm = std::getenv("COLORTERM");
  return colorterm != nullptr && strcmp(colorterm, "truecolor") == 0;
}

// Print images of at most 120 x 120 characters. If images are larger, they are
// downscaled (AREA sampling).
constexpr int kMaxCharsX = 120;
constexpr int kMaxCharsY = 120;

// Table for nice ASCII Art.
constexpr char kGrayTable[] = " .:-=+*#%@";

// Maps a value between 0 and 1 to a character in kGrayTable.
char MapToAscii(float value) {
  value = std::clamp(value, 0.0f, 1.0f);
  int int_val = static_cast<int>(strlen(kGrayTable) * value);
  return kGrayTable[std::min<int>(int_val, strlen(kGrayTable) - 1)];
}

template <typename T>
double GetNormalizedValue(const uint8_t* ptr, int idx) {
  constexpr double min = std::numeric_limits<T>::lowest();
  constexpr double max = std::numeric_limits<T>::max();
  return (reinterpret_cast<const T*>(ptr)[idx] - min) / (max - min);
}

absl::StatusOr<std::function<double(const uint8_t*, int)>>
GetMatElementAccessor(const cv::Mat& mat) {
  switch (mat.depth()) {
    case CV_8U:
      return [](const uint8_t* mat_ptr, int idx) -> double {
        return GetNormalizedValue<uint8_t>(mat_ptr, idx);
      };
    case CV_8S:
      return [](const uint8_t* mat_ptr, int idx) -> double {
        return GetNormalizedValue<int8_t>(mat_ptr, idx);
      };
    case CV_16U:
      return [](const uint8_t* mat_ptr, int idx) -> double {
        return GetNormalizedValue<uint16_t>(mat_ptr, idx);
      };
    case CV_16S:
      return [](const uint8_t* mat_ptr, int idx) -> double {
        return GetNormalizedValue<int16_t>(mat_ptr, idx);
      };
    case CV_32S:
      return [](const uint8_t* mat_ptr, int idx) -> double {
        return GetNormalizedValue<int32_t>(mat_ptr, idx);
      };
    case CV_32F:
      return [](const uint8_t* mat_ptr, int idx) -> double {
        return reinterpret_cast<const float*>(mat_ptr)[idx];
      };
    case CV_64F:
      return [](const uint8_t* mat_ptr, int idx) -> double {
        return reinterpret_cast<const double*>(mat_ptr)[idx];
      };
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unhandled mat depth ", mat.depth()));
  }
}

std::tuple<int, int, int> GetRGB(
    const std::function<double(const uint8_t*, int)>& accessor,
    const uint8_t* ptr, int x, int y, int num_channels) {
  if (!ptr) {
    return std::make_tuple(0, 0, 0);
  }

  double r, g, b;
  if (num_channels == 1) {
    r = g = b = accessor(ptr, x);
  }

  if (num_channels == 2) {
    r = accessor(ptr, x * 2 + 0);
    g = accessor(ptr, x * 2 + 1);
    b = 0.0;
  }

  if (num_channels == 3) {
    r = accessor(ptr, x * 3 + 0);
    g = accessor(ptr, x * 3 + 1);
    b = accessor(ptr, x * 3 + 2);
  }

  if (num_channels == 4) {
    r = accessor(ptr, x * 4 + 0);
    g = accessor(ptr, x * 4 + 1);
    b = accessor(ptr, x * 4 + 2);
    double a = accessor(ptr, x * 4 + 3);

    // Make a checkerboard.
    bool is_odd = (x / 2 + y / 2) & 1;
    double checker = is_odd ? 0.25 : 0.75;
    r = r * a + checker * (1.0 - a);
    g = g * a + checker * (1.0 - a);
    b = b * a + checker * (1.0 - a);
  }

  return std::make_tuple(static_cast<int>(r * 255.0),
                         static_cast<int>(g * 255.0),
                         static_cast<int>(b * 255.0));
}

using ChannelMapper = absl::AnyInvocable<float(
    const float* channel_vec, int num_input_channels, int output_channel)>;

void LogMatImpl(const cv::Mat& mat, absl::string_view name) {
  int width = mat.cols;
  int height = mat.rows;
  int num_channels = mat.channels();

  bool is_true_color_term = IsTrueColorTerm();

  // Use half as many rows than cols since ASCII chars are higher than wide.
  int small_width = std::min(kMaxCharsX, width);
  int divisor = is_true_color_term ? 1 : 2;
  int small_height = std::max(small_width * height / (divisor * width), 1);
  if (small_height > kMaxCharsY) {
    small_height = kMaxCharsY;
    small_width = small_height * width * divisor / height;
  }
  cv::Mat small_mat;
  if (small_width != width || small_height != height) {
    cv::resize(mat, small_mat, cv::Size(small_width, small_height),
               cv::INTER_AREA);
  } else {
    small_mat = mat;
  }

  // The accessor function returns a value between 0 and 1 for any data type.
  auto accessor_or = GetMatElementAccessor(small_mat);
  if (!accessor_or.ok()) {
    ABSL_LOG(WARNING) << "  <cannot print: " << accessor_or.status().message()
                      << ">";
    return;
  }
  auto accessor = *accessor_or;

  // Draw the image with a nice frame.
  std::string horizontal_bar;
  horizontal_bar.reserve(small_width * 2);
  for (int x = 0; x < small_width; ++x) {
    horizontal_bar += "\u2550";
  }
  ABSL_LOG(INFO) << "\u2554" << horizontal_bar << "\u2557 " << name;
  std::string row;
  if (is_true_color_term) {
    // Use half-blocks (\u2584) and truecolor escape codes.
    for (int y = 0; y < small_height; y += 2) {
      const uint8_t* top = small_mat.ptr<uint8_t>(y);
      const uint8_t* bottom =
          y + 1 < small_height ? small_mat.ptr<uint8_t>(y + 1) : nullptr;
      for (int x = 0; x < small_width; ++x) {
        int rt, gt, bt, rb, gb, bb;
        std::tie(rt, gt, bt) = GetRGB(accessor, top, x, y, num_channels);
        std::tie(rb, gb, bb) = GetRGB(accessor, bottom, x, y + 1, num_channels);
        row += absl::StrFormat("\033[48;2;%d;%d;%dm\033[38;2;%d;%d;%dm\u2584",
                               rt, gt, bt, rb, gb, bb);
      }
      row += "\033[0m";
      // Use name as postfix for easy log grepping.
      ABSL_LOG(INFO) << "\u2551" << row << "\u2551 " << name;
      row.clear();
    }
  } else {
    // Use ASCII art.
    for (int y = 0; y < small_height; ++y) {
      const uint8_t* mat_ptr = small_mat.ptr<uint8_t>(y);
      for (int x = 0; x < small_width; ++x) {
        double value = 0.0f;
        for (int c = 0; c < num_channels; ++c) {
          value += accessor(mat_ptr, x * num_channels + c);
        }
        row.push_back(MapToAscii(value / num_channels));
      }
      // Use name as postfix for easy log grepping.
      ABSL_LOG(INFO) << "\u2551" << row << "\u2551 " << name;
      row.clear();
    }
  }
  ABSL_LOG(INFO) << "\u255a" << horizontal_bar << "\u255d " << name;
}

void LogTensorImpl(const Tensor& tensor, float min_range, float max_range,
                   int num_output_channels, ChannelMapper mapper,
                   absl::string_view name) {
  if (tensor.element_type() != Tensor::ElementType::kFloat32) {
    ABSL_LOG(WARNING) << "  <cannot log tensor of type "
                      << static_cast<int>(tensor.element_type())
                      << ", required: float>";
    return;
  }

  int height = tensor.shape().dims[1];
  int width = tensor.shape().dims[2];
  int num_channels = tensor.shape().dims[3];

  if (tensor.shape().dims[0] == 0 || width == 0 || height == 0 ||
      num_channels == 0) {
    ABSL_LOG(INFO) << "  <empty>";
    return;
  }

  cv::Mat mat(height, width, CV_MAKETYPE(CV_8U, num_output_channels));

  Tensor::CpuReadView read_view = tensor.GetCpuReadView();
  float scale = 255.0f / (max_range - min_range);
  const float* tensor_ptr = read_view.buffer<float>();
  for (int y = 0; y < height; ++y) {
    uint8_t* row = mat.ptr<uint8_t>(y);
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < num_output_channels; ++c) {
        const float* channel_vec = tensor_ptr + (y * width + x) * num_channels;
        float value = mapper(channel_vec, num_channels, c);
        row[x * num_output_channels + c] = static_cast<uint8_t>(
            std::clamp((value - min_range) * scale, 0.0f, 255.0f));
      }
    }
  }

  LogMatImpl(mat, name);
}

}  // namespace

void LogTensorChannel(const Tensor& tensor, int channel, absl::string_view name,
                      float min_range, float max_range) {
  ABSL_LOG(INFO) << name << "[" << tensor.shape().dims << "], channel "
                 << channel << " =";

  if (tensor.shape().dims.size() != 4 || tensor.shape().dims[0] != 1 ||
      channel < 0 || channel >= tensor.shape().dims[3]) {
    ABSL_LOG(WARNING) << "  <cannot log channel " << channel
                      << " of tensor with shape " << tensor.shape().dims << ">";
    return;
  }

  LogTensorImpl(
      tensor, min_range, max_range, /*num_output_channels=*/1,
      [channel](const float* channel_vec, int num_input_channels,
                int output_channel) { return channel_vec[channel]; },
      name);
}

void LogTensor(const Tensor& tensor, absl::string_view name, float min_range,
               float max_range) {
  if (tensor.shape().dims.size() != 4 || tensor.shape().dims[0] != 1) {
    ABSL_LOG(INFO) << name << "[" << tensor.shape().dims << "] = ";
    ABSL_LOG(WARNING) << "  <cannot log tensor with shape "
                      << tensor.shape().dims << ", required: [1, h, w, c]>";
    return;
  }

  int num_channels = tensor.shape().dims.back();
  if (num_channels <= 3) {
    // Log tensor as RGB or grayscale image.
    ABSL_LOG(INFO) << name << "[" << tensor.shape().dims << "] = ";
    LogTensorImpl(
        tensor, min_range, max_range, num_channels,
        [](const float* channel_vec, int num_input_channels,
           int output_channel) { return channel_vec[output_channel]; },
        name);
  } else {
    // Log tensor channel averages as grayscale image.
    ABSL_LOG(INFO) << name << "[" << tensor.shape().dims
                   << "], channel average = ";
    LogTensorImpl(
        tensor, min_range, max_range, 1,
        [](const float* channel_vec, int num_input_channels,
           int output_channel) {
          float sum = 0.0f;
          for (int c = 0; c < num_input_channels; ++c) {
            sum += channel_vec[c];
          }
          return sum / num_input_channels;
        },
        name);
  }
}

void LogImage(const ImageFrame& image, absl::string_view name) {
  return LogMat(formats::MatView(&image), name);
}

// Logs the given mat as ASCII image.
void LogMat(const cv::Mat& mat, absl::string_view name) {
  int width = mat.cols;
  int height = mat.rows;
  int num_channels = mat.channels();

  ABSL_LOG(INFO) << name << "[" << width << " " << height << " " << num_channels
                 << "] =";

  if (width == 0 || height == 0 || num_channels == 0) {
    ABSL_LOG(INFO) << "  <empty>";
    return;
  }

  LogMatImpl(mat, name);
}

void LogHalideBuffer(Halide::Runtime::Buffer<const uint8_t> buffer,
                     absl::string_view name) {
  std::vector<int> dims(buffer.dimensions());
  for (int i = 0; i < buffer.dimensions(); ++i) {
    dims[i] = buffer.extent(i);
  }
  ABSL_LOG(INFO) << name << "[" << dims << "] =";

  if (buffer.dimensions() > 3) {
    ABSL_LOG(WARNING) << "  <cannot log Halide buffer with "
                      << buffer.dimensions() << " dimensions, required: <= 3>";
    return;
  }
  if (buffer.dimensions() == 0) {
    ABSL_LOG(INFO) << "  <empty>";
    return;
  }

  // cv::Mat only supports mapping to interleaved buffers (channels must be
  // consecutive).
  bool is_interleaved = buffer.dimensions() < 3 || buffer.stride(2) == 1;
  if (!is_interleaved) {
    buffer = buffer.copy_to_interleaved();
  }

  const int sizes[] = {buffer.height(), buffer.width()};
  const int type = CV_MAKETYPE(CV_8U, buffer.channels());
  const size_t steps[] = {
      static_cast<size_t>(buffer.dimensions() > 1 ? buffer.stride(1) : 1),
      static_cast<size_t>(buffer.stride(0))};
  cv::Mat mat(2, sizes, type, const_cast<unsigned char*>(buffer.data()), steps);
  LogMatImpl(mat, name);
}

}  // namespace mediapipe::debug

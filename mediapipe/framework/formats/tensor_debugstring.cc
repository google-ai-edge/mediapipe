#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "mediapipe/framework/formats/tensor.h"

namespace mediapipe {
namespace {

// If there are more than 8 elements, display x1 x2 x3 ... xn-2 xn-1 xn.
constexpr int kMaxUnshortenedElements = 8;
constexpr int kNumElementsBeforeAndAfterDots = 3;

// Context used for formatting tensor values into a string.
struct FormatContext {
  const std::vector<int>& dims;
  const std::vector<int>& dims_prods;
  const std::vector<bool>& dim_was_shortened;
  const std::vector<std::string>& values;
  int column_width;
  int max_digits_after_dot;
  std::stringstream& ss;
};

// Formats a single tensor dimension.
void FormatTensorDim(FormatContext& ctx, int dim_index, int start_index,
                     bool last_element_in_dim) {
  if (dim_index == ctx.dims.size()) {
    const auto& value = ctx.values[start_index];
    int len = static_cast<int>(value.size());
    int spaces_before = ctx.column_width - len;
    int spaces_after = last_element_in_dim ? 0 : 1;

    // Align the numbers so that the decimal points. If there is no decimal
    // point, shift the number to the left so that it aligns properly as well.
    int dot_pos = ctx.values[start_index].find('.');
    int num_digits_after_dot =
        dot_pos != std::string::npos
            ? static_cast<int>(value.size()) - dot_pos - 1
        : ctx.max_digits_after_dot == 0
            ? 0
            : -1;  // Ints don't have a dot, so set -1 to shift one more.
    int shift = ctx.max_digits_after_dot - num_digits_after_dot;
    spaces_before -= shift;
    spaces_after += shift;

    ABSL_DCHECK_GE(spaces_before, 0);
    ABSL_DCHECK_GE(spaces_after, 0);
    for (int n = 0; n < spaces_before; ++n) ctx.ss << " ";
    ctx.ss << value;
    for (int n = 0; n < spaces_after; ++n) ctx.ss << " ";
    return;
  }

  ctx.ss << "[";
  if (ctx.dim_was_shortened[dim_index]) {
    // Add the first and last few elements.
    ABSL_DCHECK_EQ(ctx.dims[dim_index], kNumElementsBeforeAndAfterDots * 2);
    for (int n = 0; n < kNumElementsBeforeAndAfterDots; ++n) {
      FormatTensorDim(ctx, dim_index + 1,
                      start_index + n * ctx.dims_prods[dim_index], false);
    }

    ctx.ss << "...";
    for (int n = dim_index + 1; n < ctx.dims.size(); ++n) ctx.ss << "\n";
    if (dim_index < ctx.dims.size() - 1) {
      for (int n = 0; n <= dim_index; ++n) ctx.ss << " ";
    } else {
      ctx.ss << " ";
    }

    for (int n = kNumElementsBeforeAndAfterDots; n < ctx.dims[dim_index]; ++n) {
      FormatTensorDim(ctx, dim_index + 1,
                      start_index + n * ctx.dims_prods[dim_index],
                      n == ctx.dims[dim_index] - 1);
    }
  } else {
    // Add all elements.
    for (int n = 0; n < ctx.dims[dim_index]; ++n) {
      FormatTensorDim(ctx, dim_index + 1,
                      start_index + n * ctx.dims_prods[dim_index],
                      n == ctx.dims[dim_index] - 1);
    }
  }
  ctx.ss << "]";
  if (!last_element_in_dim) {
    if (dim_index > 0) {
      for (int n = dim_index; n < ctx.dims.size(); ++n) ctx.ss << "\n";
    }
    for (int n = 0; n < dim_index; ++n) ctx.ss << " ";
  }
}

// Returns a formatted string representation of the tensor values.
std::string FormatTensorValues(const std::vector<int>& dims,
                               const std::vector<int>& dims_prods,
                               const std::vector<bool>& dim_was_shortened,
                               const std::vector<std::string>& values) {
  // Align the numbers so that the decimal points are aligned.
  int max_digits_before_dot = 0;
  int max_digits_after_dot = 0;
  for (const auto& value : values) {
    int dot_pos = value.find('.');
    if (dot_pos != std::string::npos) {
      max_digits_before_dot = std::max(max_digits_before_dot, dot_pos);
      max_digits_after_dot =
          std::max<int>(max_digits_after_dot, value.size() - dot_pos - 1);
    } else {
      max_digits_before_dot =
          std::max<int>(max_digits_before_dot, value.size());
    }
  }

  // Space reserved for a single number. Since we align all numbers by the
  // decimal point, we can't just take the max string length.
  int column_width = max_digits_after_dot > 0
                         ? max_digits_before_dot + 1 + max_digits_after_dot
                         : max_digits_before_dot;

  std::stringstream ss;
  FormatContext ctx = {dims,   dims_prods,   dim_was_shortened,
                       values, column_width, max_digits_after_dot,
                       ss};
  FormatTensorDim(ctx, /*dim_index=*/0, /*start_index=*/0, false);
  return ss.str();
}

template <typename T>
std::string FormatTensorValue(T value) {
  return absl::StrFormat("%v", value);
}

template <>
std::string FormatTensorValue<float>(float value) {
  return absl::StrFormat("%.7g", value);
}

template <>
std::string FormatTensorValue<char>(char value) {
  // Print the printable ascii chars as is and escape the rest.
  return value >= 32 && value <= 126
             ? absl::StrFormat("%c", value)
             : absl::StrFormat("\\x%02x", static_cast<uint8_t>(value));
}

// Converts tensor values to strings, taking shortened dimensions into account.
template <typename T>
void AppendValueStringsForDim(const std::vector<int>& dims,
                              const std::vector<int>& dims_prods,
                              const std::vector<bool>& dim_was_shortened,
                              int dim_index, int start_index, const T* data,
                              std::vector<std::string>& values_str) {
  if (dim_index == dims.size()) {
    values_str.push_back(FormatTensorValue(data[start_index]));
    return;
  }

  auto recurse = [&](int n) {
    AppendValueStringsForDim(dims, dims_prods, dim_was_shortened, dim_index + 1,
                             start_index + n * dims_prods[dim_index], data,
                             values_str);
  };

  if (dim_was_shortened[dim_index]) {
    // Add the first and last few elements.
    ABSL_DCHECK_GT(dims[dim_index], kNumElementsBeforeAndAfterDots * 2);
    for (int n = 0; n < kNumElementsBeforeAndAfterDots; ++n) {
      recurse(n);
    }
    for (int n = dims[dim_index] - kNumElementsBeforeAndAfterDots;
         n < dims[dim_index]; ++n) {
      recurse(n);
    }
  } else {
    // Add all elements.
    for (int n = 0; n < dims[dim_index]; ++n) {
      recurse(n);
    }
  }
}

// Returns a string representation of the typed tensor values.
template <typename T>
std::string ValuesStringT(int max_num_elements, int num_elements,
                          const std::vector<int>& dims, const T* data) {
  // If true, a dim is displayed with a "..." between the first and last few
  // elements, e.g. [1, 2, 3, ..., 97, 98, 99].
  std::vector<bool> dim_was_shortened(dims.size(), false);

  // Compute dim prods, e.g. for [2, 3, 4], compute [12, 4, 1].
  std::vector<int> dims_prods(dims.size(), 1);
  for (int n = dims.size() - 2; n >= 0; --n) {
    dims_prods[n] = dims_prods[n + 1] * dims[n + 1];
  }

  // Shorten dimensions in case num_elements > max_num_elements (we print ...).
  std::vector<int> shortened_dims = dims;

  std::vector<std::string> values_str;
  if (num_elements <= max_num_elements) {
    // Print all elements.
    values_str.reserve(num_elements);
    for (int n = 0; n < num_elements; ++n) {
      values_str.push_back(FormatTensorValue(data[n]));
    }
  } else {
    // Shorten dimensions with more than kMaxUnshortenedElements elements.
    int num_shortened_elements = 1;
    for (int n = 0; n < dims.size(); ++n) {
      if (dims[n] > kMaxUnshortenedElements) {
        shortened_dims[n] = kNumElementsBeforeAndAfterDots * 2;
        dim_was_shortened[n] = true;
      }
      num_shortened_elements *= shortened_dims[n];
    }
    values_str.reserve(num_shortened_elements);
    AppendValueStringsForDim(dims, dims_prods, dim_was_shortened,
                             /*dim_index=*/0,
                             /*start_index=*/0, data, values_str);
    ABSL_DCHECK_EQ(values_str.size(), num_shortened_elements);
  }

  std::vector<int> shortened_dims_prods(shortened_dims.size(), 1);
  for (int n = shortened_dims.size() - 2; n >= 0; --n) {
    shortened_dims_prods[n] =
        shortened_dims_prods[n + 1] * shortened_dims[n + 1];
  }

  return FormatTensorValues(shortened_dims, shortened_dims_prods,
                            dim_was_shortened, values_str);
}

// Returns a string representation of the tensor values. If the tensor has more
// than max_num_elements elements, all dimensions larger than 8 are shortened to
// x1 x2 x3 ... xn-2, xn-1, xn.
std::string ValuesString(const Tensor& tensor, int max_num_elements) {
  const int64_t num_elements = tensor.shape().num_elements();
  if (num_elements == 0) return "[]";

  const auto& dims = tensor.shape().dims;
  auto view = tensor.GetCpuReadView();
  const void* data = view.buffer<void>();
  switch (tensor.element_type()) {
    case Tensor::ElementType::kNone:
      return "<invalid>";
      break;
    case Tensor::ElementType::kFloat16:
      // We currently don't ship an official half class with Mediapipe. There is
      // one in mediapipe/gpu/gl_ssbo_rgba32f_to_texture_test.cc,
      // which could potentially be reused here.
      return "<printing data type not supported>";
      break;
    case Tensor::ElementType::kFloat32:
      return ValuesStringT(max_num_elements, num_elements, dims,
                           reinterpret_cast<const float*>(data));
      break;
    case Tensor::ElementType::kUInt8:
      return ValuesStringT<uint8_t>(max_num_elements, num_elements, dims,
                                    reinterpret_cast<const uint8_t*>(data));
      break;
    case Tensor::ElementType::kInt8:
      return ValuesStringT<int8_t>(max_num_elements, num_elements, dims,
                                   reinterpret_cast<const int8_t*>(data));
      break;
    case Tensor::ElementType::kInt32:
      return ValuesStringT<int32_t>(max_num_elements, num_elements, dims,
                                    reinterpret_cast<const int32_t*>(data));
      break;
    case Tensor::ElementType::kInt64:
      return ValuesStringT<int64_t>(max_num_elements, num_elements, dims,
                                    reinterpret_cast<const int64_t*>(data));
      break;
    case Tensor::ElementType::kChar:
      return ValuesStringT<char>(max_num_elements, num_elements, dims,
                                 reinterpret_cast<const char*>(data));
      break;
    case Tensor::ElementType::kBool:
      return ValuesStringT<bool>(max_num_elements, num_elements, dims,
                                 reinterpret_cast<const bool*>(data));
      break;
  }
}

}  // namespace

std::string Tensor::DebugString(int max_num_elements) const {
  return absl::StrCat("Tensor<", ElementTypeName(element_type()), "> [",
                      absl::StrJoin(shape().dims, " "), "] =\n",
                      ValuesString(*this, max_num_elements));
}

}  // namespace mediapipe

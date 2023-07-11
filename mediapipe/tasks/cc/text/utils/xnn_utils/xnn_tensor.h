#ifndef MEDIAPIPE_TASKS_CC_TEXT_UTILS_XNN_UTILS_XNN_TENSOR_H_
#define MEDIAPIPE_TASKS_CC_TEXT_UTILS_XNN_UTILS_XNN_TENSOR_H_

#include <fcntl.h>
#include <sys/mman.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "file/base/helpers.h"
#include "file/base/options.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/text/utils/xnn_utils/utils.h"
#include "third_party/XNNPACK/include/xnnpack.h"
#include "util/gtl/stl_logging.h"

namespace mediapipe {
namespace xnn_utils {

static constexpr absl::string_view kQuantizedScaleSuffix{"_quantized_scale"};
static constexpr absl::string_view kSparsityParamsSuffix{"_sparsity_params"};

struct Tensor {
  using DimsType = std::vector<size_t>;

  explicit Tensor(DimsType in_dims, xnn_datatype datatype_ = xnn_datatype_fp32)
      : dims(std::move(in_dims)),
        num_elements(dims.empty() ? 0
                                  : std::accumulate(std::begin(dims),
                                                    std::end(dims), size_t(1),
                                                    std::multiplies<size_t>())),
        datatype(datatype_) {}
  Tensor(Tensor&& other) = default;

  Tensor& operator=(const Tensor& other) = delete;
  Tensor& operator=(Tensor&& other) = default;

  virtual ~Tensor() = default;

  bool operator==(const Tensor& other) const;

  void SetMetadata(absl::string_view key, int value) { metadata[key] = value; }

  std::optional<int> GetMetadata(absl::string_view key) const {
    if (metadata.contains(key)) {
      return metadata.at(key);
    }
    return std::nullopt;
  }

  // Read weights from file.
  template <xnn_datatype xnn_datatype_ = xnn_datatype_fp32>
  static absl::StatusOr<std::shared_ptr<Tensor>> FromFile(
      absl::string_view file_path, DimsType dims, bool use_mmap = true) {
    auto result = std::make_shared<Tensor>(std::move(dims), xnn_datatype_);

    MP_RETURN_IF_ERROR(
        result->LoadFromFile(file_path, use_mmap, /*exact_match=*/true));

    return result;
  }

  virtual absl::Status DefineAsExternal(xnn_subgraph& subgraph, uint32_t flags);
  absl::Status DefineAsInput(xnn_subgraph& subgraph);
  absl::Status DefineAsOutput(xnn_subgraph& subgraph);
  absl::Status DefineAsIntermediateTensor(xnn_subgraph& subgraph);
  virtual absl::Status DefineWeight(xnn_subgraph& subgraph, uint32_t flags);
  absl::Status DefineWeight(xnn_subgraph& subgraph);
  absl::Status DefineRope(xnn_subgraph& subgraph);

  absl::Status LoadFromBuffer(const void* buffer);
  absl::Status LoadFromVec(const std::vector<float>& data,
                           bool exact_match = true);
  absl::Status LoadFromVec(std::vector<float>&& data, bool exact_match = true);
  absl::Status LoadFromFile(absl::string_view file_path) {
    return LoadFromFile(file_path, true, true);
  }
  virtual absl::Status LoadFromFile(absl::string_view file_path, bool use_mmap,
                                    bool exact_match);

  absl::Status DumpToBuffer(void* buffer);
  absl::Status DumpToVec(std::vector<float>& out_data, bool exact_match = true);
  virtual absl::Status DumpToFile(absl::string_view file_path);

  // If ith offset is 0, view's ith dim equals to original ith dim, otherwise 1.
  std::shared_ptr<Tensor> Slice(DimsType offset);
  // Slice along the `index`th dimension, offset at this dimension.
  std::shared_ptr<Tensor> Slice(size_t index, size_t offset);

  // Point the underline data to the borrowed tensor's data.
  Tensor& Borrow(std::shared_ptr<Tensor>, size_t element_offset = 0);
  std::shared_ptr<Tensor> View();
  virtual std::shared_ptr<Tensor> View(DimsType as_dims,
                                       size_t dim_scale_if_any = 0);

  Tensor& MarkOutput() {
    AllocateBufferIfNeeded();
    is_output_tensor = true;
    return *this;
  }

  virtual void* Data();
  const void* Data() const;

  template <typename T>
  T* DataAs() {
    DCHECK_EQ(ElementSize(), sizeof(T));
    return static_cast<T*>(Data());
  }
  template <typename T>
  const T* DataAs() const {
    return static_cast<const T*>(Data());
  }

  virtual std::shared_ptr<Tensor> Transpose();

  virtual absl::StatusOr<std::shared_ptr<Tensor>> ConvertToF32();

  DimsType dims;
  size_t num_elements = 0;
  xnn_datatype datatype = xnn_datatype_invalid;
  uint32_t tensor_id = XNN_INVALID_VALUE_ID;

  // shared_ptr to make TensorMetadata copyable.
  std::shared_ptr<char> flat_data;

 protected:
  friend class XnnGraphBuilder;
  friend class XnnGraph;

  // Actually allocate buffer unless necessary.
  virtual void AllocateBufferIfNeeded();

  virtual size_t ElementSize() const { return 4; }

  bool is_output_tensor = false;

  absl::flat_hash_map<std::string, int> metadata;
};

std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

// Channelwise Quantized.
struct QCTensor : public Tensor {
  explicit QCTensor(DimsType in_dims, size_t dim_scale_if_any)
      : Tensor(std::move(in_dims)), dim_scale(dim_scale_if_any) {
    datatype = xnn_datatype_qcint8;
    CHECK_LT(dim_scale, 4);
  }

  void AllocateBufferIfNeeded() override;
  size_t ElementSize() const override { return 1; }

  virtual absl::Status LoadFromFile(absl::string_view quantized_weight_filename,
                                    absl::string_view scale_filename,
                                    bool use_mmap, bool exact_match);
  // Append kQuantizedScaleSuffix to use as scale filename.
  absl::Status LoadFromFile(absl::string_view file_path, bool use_mmap,
                            bool exact_match) override {
    return LoadFromFile(file_path,
                        absl::StrCat(file_path, kQuantizedScaleSuffix),
                        use_mmap, exact_match);
  }

  absl::Status DumpToFile(absl::string_view file_path) override;

  absl::Status DefineWeight(xnn_subgraph& subgraph, uint32_t flags) override;

  std::shared_ptr<Tensor> Transpose() override;

  absl::StatusOr<std::shared_ptr<Tensor>> ConvertToF32() override;

  std::shared_ptr<Tensor> View(DimsType as_dims,
                               size_t dim_scale_if_any) override;

  std::shared_ptr<float> scale_data;
  // Index of the dimension to scale.
  size_t dim_scale;
};

std::ostream& operator<<(std::ostream& os, const QCTensor& tensor);

absl::Status FillXnnRoPEWeights(Tensor& out_seg_pos);

}  // namespace xnn_utils
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_TEXT_UTILS_XNN_UTILS_XNN_TENSOR_H_

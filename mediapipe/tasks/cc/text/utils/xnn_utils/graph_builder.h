#ifndef MEDIAPIPE_TASKS_CC_TEXT_UTILS_XNN_UTILS_GRAPH_BUILDER_H_
#define MEDIAPIPE_TASKS_CC_TEXT_UTILS_XNN_UTILS_GRAPH_BUILDER_H_

#include <sys/types.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/source_location.h"
#include "file/base/helpers.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/text/utils/xnn_utils/xnn_tensor.h"
#include "third_party/XNNPACK/include/xnnpack.h"

namespace mediapipe {
namespace xnn_utils {

using XnnSubgraphPtr =
    std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)>;
using XnnRuntimePtr =
    std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)>;
using XnnThreadpoolPtr =
    std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)>;

struct ClampParams {
  float out_min = -std::numeric_limits<float>::infinity();
  float out_max = std::numeric_limits<float>::infinity();
};

struct FullConnParams : public ClampParams {
  bool transpose = false;
};

struct RuntimeConfigs {
  bool xnn_profile;
  std::string xnn_profile_csv;
  size_t xnn_num_threads;
};

class XnnGraph;

// XnnGraphBuilder is used to construct XnnGraph (through Build()). Once a
// XnnGraph is constructed, it can run for multiple times.
class XnnGraphBuilder {
 public:
  static constexpr absl::string_view kKeySelfAttentionReshapedWeight{
      "self_attention_reshaped_weight_N"};

  explicit XnnGraphBuilder(xnn_datatype data_type = xnn_datatype_fp32)
      : data_type_(data_type) {}
  virtual ~XnnGraphBuilder() = default;

  absl::StatusOr<std::unique_ptr<XnnGraph>> Build(
      std::unique_ptr<RuntimeConfigs> runtime_configs = nullptr);

  // New input or output tensor.
  absl::StatusOr<std::shared_ptr<Tensor>> NewInput(
      Tensor::DimsType dims,
      absl::SourceLocation loc = absl::SourceLocation::current());

  // New static weight, populate value before Build()
  absl::StatusOr<std::shared_ptr<Tensor>> NewWeight(
      Tensor::DimsType dims,
      absl::SourceLocation loc = absl::SourceLocation::current());
  absl::StatusOr<std::shared_ptr<Tensor>> NewWeight(
      absl::string_view file_path, Tensor::DimsType dims,
      absl::SourceLocation loc = absl::SourceLocation::current());
  void NewWeight(std::shared_ptr<Tensor> t,
                 absl::SourceLocation loc = absl::SourceLocation::current());

  // Element wise square.
  absl::StatusOr<std::shared_ptr<Tensor>> Square(
      std::shared_ptr<Tensor> input,
      absl::SourceLocation loc = absl::SourceLocation::current());

  absl::StatusOr<std::shared_ptr<Tensor>> SquareRoot(
      std::shared_ptr<Tensor> input,
      absl::SourceLocation loc = absl::SourceLocation::current());

  absl::StatusOr<std::shared_ptr<Tensor>> Gelu(
      std::shared_ptr<Tensor> input,
      absl::SourceLocation loc = absl::SourceLocation::current());

  absl::StatusOr<std::shared_ptr<Tensor>> Clamp(
      std::shared_ptr<Tensor> input, ClampParams params,
      absl::SourceLocation loc = absl::SourceLocation::current());

  absl::StatusOr<std::shared_ptr<Tensor>> Tanh(
      std::shared_ptr<Tensor> input,
      absl::SourceLocation loc = absl::SourceLocation::current());

  // logits = cap * jnp.tanh(logits / cap)
  absl::StatusOr<std::shared_ptr<Tensor>> CapTanh(
      std::shared_ptr<Tensor> input, float cap,
      absl::SourceLocation loc = absl::SourceLocation::current());

  // Average over last dimension, keep num of dims same.
  absl::StatusOr<std::shared_ptr<Tensor>> AvgLastDim(
      std::shared_ptr<Tensor> input,
      absl::SourceLocation loc = absl::SourceLocation::current());

  absl::StatusOr<std::shared_ptr<Tensor>> Rms(
      std::shared_ptr<Tensor> input,
      absl::SourceLocation loc = absl::SourceLocation::current());

  absl::StatusOr<std::shared_ptr<Tensor>> RmsNorm(
      std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> scale,
      absl::SourceLocation loc = absl::SourceLocation::current());

  absl::StatusOr<std::shared_ptr<Tensor>> Reshape(
      std::shared_ptr<Tensor> input, Tensor::DimsType new_dims,
      absl::SourceLocation loc = absl::SourceLocation::current());

  absl::StatusOr<std::shared_ptr<Tensor>> Permute(
      std::shared_ptr<Tensor> input, Tensor::DimsType permute,
      absl::SourceLocation loc = absl::SourceLocation::current());

  // input: [B * I]
  // filter: [O * I], [I * O] if transpose
  // return: [B * O]
  absl::StatusOr<std::shared_ptr<Tensor>> MatMul(
      std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight,
      absl::SourceLocation loc = absl::SourceLocation::current()) {
    return MatMul(input, weight, FullConnParams(), loc);
  }

  absl::StatusOr<std::shared_ptr<Tensor>> MatMul(
      std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight,
      FullConnParams params,
      absl::SourceLocation loc = absl::SourceLocation::current()) {
    return FullConn(input, weight, nullptr, params, loc);
  }

  absl::StatusOr<std::shared_ptr<Tensor>> BatchMatMul(
      std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight,
      FullConnParams params = FullConnParams(),
      absl::SourceLocation loc = absl::SourceLocation::current());

  absl::StatusOr<std::shared_ptr<Tensor>> FullConn(
      std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight,
      std::shared_ptr<Tensor> bias,
      absl::SourceLocation loc = absl::SourceLocation::current()) {
    return FullConn(input, weight, bias, FullConnParams(), loc);
  }

  absl::StatusOr<std::shared_ptr<Tensor>> FullConn(
      std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight,
      std::shared_ptr<Tensor> bias, FullConnParams params,
      absl::SourceLocation loc = absl::SourceLocation::current());

  absl::StatusOr<std::shared_ptr<Tensor>> Softmax(
      std::shared_ptr<Tensor> input,
      absl::SourceLocation loc = absl::SourceLocation::current());

  absl::StatusOr<std::shared_ptr<Tensor>> SelfAttentionProj(
      std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight,
      absl::SourceLocation loc = absl::SourceLocation::current());

  absl::StatusOr<std::shared_ptr<Tensor>> ElementAdd(
      std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs,
      ClampParams params = ClampParams(),
      absl::SourceLocation loc = absl::SourceLocation::current());

  absl::StatusOr<std::shared_ptr<Tensor>> ElementAdd(
      std::shared_ptr<Tensor> lhs, float rhs,
      ClampParams params = ClampParams(),
      absl::SourceLocation loc = absl::SourceLocation::current());

  absl::StatusOr<std::shared_ptr<Tensor>> ElementMul(
      std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs,
      ClampParams params = ClampParams(),
      absl::SourceLocation loc = absl::SourceLocation::current());

  absl::StatusOr<std::shared_ptr<Tensor>> ElementMul(
      std::shared_ptr<Tensor> lhs, float rhs,
      ClampParams params = ClampParams(),
      absl::SourceLocation loc = absl::SourceLocation::current());

  absl::StatusOr<std::shared_ptr<Tensor>> ElementDiv(
      std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs,
      ClampParams params = ClampParams(),
      absl::SourceLocation loc = absl::SourceLocation::current());

  absl::StatusOr<std::shared_ptr<Tensor>> ElementDiv(
      std::shared_ptr<Tensor> lhs, float rhs,
      ClampParams params = ClampParams(),
      absl::SourceLocation loc = absl::SourceLocation::current());

  absl::StatusOr<std::shared_ptr<Tensor>> Rope(
      std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> segment_pos,
      absl::SourceLocation loc = absl::SourceLocation::current());

  absl::StatusOr<std::shared_ptr<Tensor>> PerDimScale(
      std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> per_dim_scale,
      absl::SourceLocation loc = absl::SourceLocation::current());

  absl::StatusOr<std::shared_ptr<Tensor>> DotAttention(
      std::shared_ptr<Tensor> query_proj, std::shared_ptr<Tensor> key_proj,
      std::shared_ptr<Tensor> value_proj, std::shared_ptr<Tensor> atten_mask,
      std::shared_ptr<Tensor> per_dim_scale,
      absl::SourceLocation loc = absl::SourceLocation::current());

 protected:
  absl::StatusOr<std::shared_ptr<Tensor>> IntermediateTensor(
      Tensor::DimsType dims,
      absl::SourceLocation loc = absl::SourceLocation::current());

  const xnn_datatype data_type_;

  std::vector<std::pair<absl::SourceLocation,
                        std::function<absl::Status(xnn_subgraph_t)>>>
      build_steps_;

  absl::flat_hash_set<std::shared_ptr<Tensor>> input_tensors_;
  absl::flat_hash_set<std::shared_ptr<Tensor>> interm_tensors_;

  // TODO: fix this.
  // This is sort of bug that the weights used for rope has to be defined with
  // EXTERNAL flag, but with id out of the external range.
  absl::flat_hash_set<std::shared_ptr<Tensor>> rope_weigths_;

  // Caches
  absl::flat_hash_map<
      size_t /*dim*/,
      absl::flat_hash_map<const Tensor* /*scale*/, std::shared_ptr<Tensor>>>
      per_dim_scale_cache_;
};

class XnnGraph {
 public:
  XnnGraph(XnnSubgraphPtr subgraph,
           std::unique_ptr<RuntimeConfigs> runtime_configs)
      : owned_subgraph_(std::move(subgraph)),
        runtime_configs_(std::move(runtime_configs)) {
    DCHECK(runtime_configs_);
  }
  XnnGraph(XnnGraph&& other) = default;
  virtual ~XnnGraph() = default;

  // xnn_subgraph should be created with same size.
  virtual absl::Status Run();

 protected:
  friend class XnnGraphBuilder;

  absl::Status CreateRuntime();
  absl::Status SetupRuntime();

  XnnSubgraphPtr owned_subgraph_;

  absl::flat_hash_map<size_t, Tensor> avg_cache_;
  absl::flat_hash_map<size_t, Tensor> cap_tanh_cache_;

  // Runtime
  std::unique_ptr<RuntimeConfigs> runtime_configs_;
  XnnRuntimePtr runtime_{nullptr, xnn_delete_runtime};
  std::vector<xnn_external_value> externals_;

  XnnThreadpoolPtr threadpool_{nullptr, pthreadpool_destroy};

  absl::flat_hash_set<std::shared_ptr<Tensor>> input_tensors_;
  absl::flat_hash_set<std::shared_ptr<Tensor>> output_tensors_;
  // TODO: see above
  absl::flat_hash_set<std::shared_ptr<Tensor>> rope_weigths_;

  absl::flat_hash_set<std::shared_ptr<Tensor>> interm_tensors_;
};

}  // namespace xnn_utils
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_TEXT_UTILS_XNN_UTILS_GRAPH_BUILDER_H_

#include "mediapipe/tasks/cc/text/utils/xnn_utils/graph_builder.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/source_location.h"
#include "file/base/helpers.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/text/utils/xnn_utils/xnn_tensor.h"
#include "third_party/XNNPACK/include/xnnpack.h"
#include "util/gtl/stl_logging.h"

namespace mediapipe {
namespace xnn_utils {
namespace {

// XNNPACK supports broadcasting, this function inferences the output shape
// based on input tensor shapes.
std::vector<size_t> OutDimsForElementwiseOp(const Tensor& lhs,
                                            const Tensor& rhs) {
  DCHECK(!lhs.dims.empty());
  DCHECK(!rhs.dims.empty());
  std::vector<size_t> lhs_dims_rev(lhs.dims.rbegin(), lhs.dims.rend());
  std::vector<size_t> rhs_dims_rev(rhs.dims.rbegin(), rhs.dims.rend());
  DCHECK([&]() -> bool {
    for (size_t i = 0; i < std::min(lhs_dims_rev.size(), rhs_dims_rev.size());
         ++i) {
      if ((lhs_dims_rev[i] != rhs_dims_rev[i]) && (lhs_dims_rev[i] != 1) &&
          (rhs_dims_rev[i] != 1)) {
        return false;
      }
    }
    return true;
  }()) << "lhs "
       << lhs.dims << " rhs " << rhs.dims;
  std::vector<size_t> out_dims(
      std::max(lhs_dims_rev.size(), rhs_dims_rev.size()));
  for (int i = 0; i < out_dims.size(); ++i) {
    if (lhs_dims_rev.size() <= i) {
      out_dims[i] = rhs_dims_rev[i];
    } else if (rhs_dims_rev.size() <= i) {
      out_dims[i] = lhs_dims_rev[i];
    } else {
      out_dims[i] = lhs_dims_rev[i] == 1 ? rhs_dims_rev[i] : lhs_dims_rev[i];
    }
  }
  return std::vector<size_t>(out_dims.rbegin(), out_dims.rend());
}

// If out_id is invalid, we need to allocate tensor for intermediate result.
// Otherwise, set out_id in out_metadata.
absl::Status MaybeAllocateIntermediateTensor(xnn_subgraph_t subgraph,
                                             uint32_t out_id,
                                             Tensor& out_metadata) {
  RET_CHECK_GT(out_metadata.dims.size(), 0);
  if (out_id == XNN_INVALID_VALUE_ID) {
    // The output is intermediate, thus allocate tensor.
    MP_RETURN_IF_ERROR(out_metadata.DefineAsIntermediateTensor(*subgraph));
  } else {
    out_metadata.tensor_id = out_id;
  }

  return absl::OkStatus();
}

absl::Status MaybeAllocateIntermediateTensor(xnn_subgraph_t subgraph,
                                             Tensor& out_metadata) {
  return MaybeAllocateIntermediateTensor(subgraph, out_metadata.tensor_id,
                                         out_metadata);
}

absl::Status AllocateIntermediateTensor(xnn_subgraph_t subgraph,
                                        Tensor& out_metadata) {
  return MaybeAllocateIntermediateTensor(subgraph, XNN_INVALID_VALUE_ID,
                                         out_metadata);
}

// 1.0/jax.nn.softplus(0.0) = 1.442695041
// scale = softplus(w) * 1.442695041 / np.sqrt(query.shape[-1])
void SoftPlus(size_t cnt, const std::vector<size_t>& query_dims, float* weight,
              float* scale) {
  constexpr double r_softplus_0 = 1.442695041;
  // softplus(x) = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
  // scale = softplus(per_dim_scale) / (sqrt(input.dims[-1]) * softplus(0))
  const double r_softplus_0_over_sqrt_d =
      r_softplus_0 / std::sqrt(query_dims.back());
  for (int i = 0; i < cnt; ++i) {
    scale[i] = log1p(exp(-abs(weight[i]))) + fmax(weight[i], 0.0f);
    scale[i] *= r_softplus_0_over_sqrt_d;
  }
}

}  // namespace

absl::StatusOr<std::unique_ptr<XnnGraph>> XnnGraphBuilder::Build(
    std::unique_ptr<RuntimeConfigs> runtime_configs) {
  if (!runtime_configs) {
    runtime_configs = std::make_unique<RuntimeConfigs>();
    runtime_configs->xnn_num_threads = 1;
    runtime_configs->xnn_profile = false;
  }
  VLOG(2) << "XnnGraphBuilder::Build() building...";
  auto build_begin = absl::Now();
  RET_CHECK_EQ(xnn_status_success, xnn_initialize(nullptr));

  absl::flat_hash_set<std::shared_ptr<Tensor>> output_tensors;
  {
    uint32_t cnt = input_tensors_.size();
    for (auto& t : interm_tensors_) {
      if (t->is_output_tensor) {
        RET_CHECK_EQ(t->tensor_id, XNN_INVALID_VALUE_ID);
        t->tensor_id = cnt++;
        output_tensors.insert(t);
      }
    }
    for (auto& t : output_tensors) {
      interm_tensors_.erase(t);
    }
    for (auto& t : rope_weigths_) {
      interm_tensors_.erase(t);
      t->tensor_id = cnt++;
    }
  }

  xnn_subgraph_t subgraph_ptr = nullptr;
  RET_CHECK_EQ(xnn_status_success,
               xnn_create_subgraph(
                   /*external_value_ids=*/input_tensors_.size() +
                       output_tensors.size() + rope_weigths_.size(),
                   /*flags=*/0, &subgraph_ptr));
  RET_CHECK_NE(subgraph_ptr, nullptr);

  XnnSubgraphPtr subgraph{subgraph_ptr, xnn_delete_subgraph};

  for (auto& input : input_tensors_) {
    MP_RETURN_IF_ERROR(input->DefineAsInput(*subgraph));
  }
  for (auto& output : output_tensors) {
    MP_RETURN_IF_ERROR(output->DefineAsOutput(*subgraph));
  }
  {
    for (auto& t : rope_weigths_) {
      MP_RETURN_IF_ERROR(t->DefineRope(*subgraph));
    }
  }

  for (auto& [loc, step] : build_steps_) {
    if (auto s = step(subgraph.get()); !s.ok()) {
      s.AddSourceLocation(loc);
      return s;
    }
  }

  XnnGraph result(std::move(subgraph), std::move(runtime_configs));
  result.input_tensors_ = std::move(input_tensors_);
  result.output_tensors_ = std::move(output_tensors);
  result.interm_tensors_ = std::move(interm_tensors_);

  VLOG(2) << "XnnGraphBuilder::Build() creating runtime...";
  auto create_begin = absl::Now();
  MP_RETURN_IF_ERROR(result.CreateRuntime());
  VLOG(2) << "XnnGraphBuilder::Build() setting up runtime...";
  auto setup_begin = absl::Now();
  MP_RETURN_IF_ERROR(result.SetupRuntime());

  auto end = absl::Now();
  VLOG(2) << "XnnGraphBuilder::Build() done build, Total " << end - build_begin
          << ", create runtime " << setup_begin - create_begin
          << ", setup runtime " << end - setup_begin;
  return std::make_unique<XnnGraph>(std::move(result));
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::NewInput(
    Tensor::DimsType dims, absl::SourceLocation loc) {
  auto t = std::make_shared<Tensor>(std::move(dims), data_type_);
  t->AllocateBufferIfNeeded();
  t->tensor_id = input_tensors_.size();
  input_tensors_.insert(t);
  return t;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::NewWeight(
    absl::string_view file_path, Tensor::DimsType dims,
    absl::SourceLocation loc) {
  ASSIGN_OR_RETURN(auto t, NewWeight(std::move(dims)));
  MP_RETURN_IF_ERROR(t->LoadFromFile(file_path));
  return t;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::NewWeight(
    Tensor::DimsType dims, absl::SourceLocation loc) {
  auto t = std::make_shared<Tensor>(std::move(dims), data_type_);
  NewWeight(t, loc);
  return t;
}

void XnnGraphBuilder::NewWeight(std::shared_ptr<Tensor> t,
                                absl::SourceLocation loc) {
  build_steps_.push_back(
      {loc, [this, t](xnn_subgraph_t subgraph) -> absl::Status {
         if (interm_tensors_.contains(t)) {
           MP_RETURN_IF_ERROR(t->DefineWeight(*subgraph));
         }
         return absl::OkStatus();
       }});

  interm_tensors_.insert(t);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::IntermediateTensor(
    Tensor::DimsType dims, absl::SourceLocation loc) {
  auto t = std::make_shared<Tensor>(std::move(dims), data_type_);

  build_steps_.push_back(
      {loc, [this, t](xnn_subgraph_t subgraph) -> absl::Status {
         // Could be moved to output tensors, thus need check.
         if (interm_tensors_.contains(t)) {
           return AllocateIntermediateTensor(subgraph, *t);
         }
         return absl::OkStatus();
       }});

  interm_tensors_.insert(t);
  return t;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Reshape(
    std::shared_ptr<Tensor> input, Tensor::DimsType new_dims,
    absl::SourceLocation loc) {
  ASSIGN_OR_RETURN(auto output, IntermediateTensor(std::move(new_dims)));
  RET_CHECK_EQ(input->num_elements, output->num_elements)
      << "otherwise reshape does not make sense.";

  build_steps_.push_back(
      {loc, [this, input, output](xnn_subgraph_t subgraph) -> absl::Status {
         MP_RETURN_IF_ERROR(MaybeAllocateIntermediateTensor(
             subgraph, output->tensor_id, *output));

         RET_CHECK_EQ(xnn_status_success,
                      xnn_define_static_reshape(
                          subgraph, output->dims.size(), output->dims.data(),
                          input->tensor_id, output->tensor_id, /*flags=*/0));
         return absl::OkStatus();
       }});
  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::FullConn(
    std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight,
    std::shared_ptr<Tensor> bias, FullConnParams params,
    absl::SourceLocation loc) {
  const auto& input_dim = input->dims;
  const auto& weight_dim = weight->dims;
  DCHECK_GT(input_dim.size(), 1);
  DCHECK_GE(weight_dim.size(), 2);
  if (weight_dim.size() == 3) {
    RET_CHECK_EQ(weight_dim[0], 1);
  } else if (weight_dim.size() == 4) {
    RET_CHECK_EQ(weight_dim[0], 1);
    RET_CHECK_EQ(weight_dim[1], 1);
  }
  if (bias) {
    RET_CHECK_LE(bias->dims.size(), 1);
  }

  Tensor::DimsType out_dims = input_dim;
  // Not considering reshape 2D
  if (params.transpose) {
    RET_CHECK_EQ(weight_dim.size(), 2) << "otherwise change following line";
    RET_CHECK_EQ(input_dim.back(), *(weight_dim.end() - 2));
    out_dims.back() = weight_dim.back();
  } else {
    RET_CHECK_EQ(input_dim.back(), weight_dim.back());
    out_dims.pop_back();
    for (size_t i = 0; i < weight_dim.size() - 1; ++i) {
      // NHD . BTD -> NHBT
      out_dims.push_back(weight_dim[i]);
    }
  }
  ASSIGN_OR_RETURN(auto output, IntermediateTensor(std::move(out_dims)));

  build_steps_.push_back(
      {loc,
       [this, input, weight, bias, params,
        output](xnn_subgraph_t subgraph) -> absl::Status {
         MP_RETURN_IF_ERROR(MaybeAllocateIntermediateTensor(
             subgraph, output->tensor_id, *output));

         RET_CHECK_EQ(
             xnn_status_success,
             xnn_define_fully_connected(
                 subgraph, params.out_min, params.out_max, input->tensor_id,
                 weight->tensor_id,
                 bias ? bias->tensor_id : XNN_INVALID_VALUE_ID,
                 output->tensor_id,
                 /*flags=*/params.transpose ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0));

         return absl::OkStatus();
       }});
  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Permute(
    std::shared_ptr<Tensor> input, Tensor::DimsType permute,
    absl::SourceLocation loc) {
  RET_CHECK_EQ(input->dims.size(), permute.size());
  const auto& old_dims = input->dims;
  std::vector<size_t> new_dims;
  for (size_t i = 0; i < permute.size(); ++i) {
    new_dims.push_back(old_dims[permute[i]]);
  }
  ASSIGN_OR_RETURN(auto output, IntermediateTensor(std::move(new_dims)));

  build_steps_.push_back(
      {loc,
       [this, permute, input, output](xnn_subgraph_t subgraph) -> absl::Status {
         MP_RETURN_IF_ERROR(MaybeAllocateIntermediateTensor(subgraph, *output));

         RET_CHECK_EQ(xnn_status_success,
                      xnn_define_static_transpose(
                          subgraph, permute.size(), permute.data(),
                          input->tensor_id, output->tensor_id, /*flags=*/0));
         return absl::OkStatus();
       }});
  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Square(
    std::shared_ptr<Tensor> input, absl::SourceLocation loc) {
  ASSIGN_OR_RETURN(auto output, IntermediateTensor(input->dims));

  build_steps_.push_back(
      {loc, [this, output, input](xnn_subgraph_t subgraph) -> absl::Status {
         MP_RETURN_IF_ERROR(MaybeAllocateIntermediateTensor(
             subgraph, output->tensor_id, *output));
         RET_CHECK_EQ(
             xnn_status_success,
             xnn_define_square(subgraph, input->tensor_id, output->tensor_id,
                               /*flags=*/0));
         return absl::Status();
       }});

  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Softmax(
    std::shared_ptr<Tensor> input, absl::SourceLocation loc) {
  ASSIGN_OR_RETURN(auto output, IntermediateTensor(input->dims));

  build_steps_.push_back(
      {loc, [this, output, input](xnn_subgraph_t subgraph) -> absl::Status {
         MP_RETURN_IF_ERROR(MaybeAllocateIntermediateTensor(
             subgraph, output->tensor_id, *output));
         RET_CHECK_EQ(
             xnn_status_success,
             xnn_define_softmax(subgraph, input->tensor_id, output->tensor_id,
                                /*flags=*/0));
         return absl::Status();
       }});

  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::SquareRoot(
    std::shared_ptr<Tensor> input, absl::SourceLocation loc) {
  ASSIGN_OR_RETURN(auto output, IntermediateTensor(input->dims));

  build_steps_.push_back(
      {loc, [this, output, input](xnn_subgraph_t subgraph) -> absl::Status {
         MP_RETURN_IF_ERROR(MaybeAllocateIntermediateTensor(
             subgraph, output->tensor_id, *output));
         RET_CHECK_EQ(xnn_status_success,
                      xnn_define_square_root(subgraph, input->tensor_id,
                                             output->tensor_id,
                                             /*flags=*/0));
         return absl::Status();
       }});

  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::AvgLastDim(
    std::shared_ptr<Tensor> input, absl::SourceLocation loc) {
  ASSIGN_OR_RETURN(auto before_reshape,
                   IntermediateTensor(Tensor::DimsType{input->dims.begin(),
                                                       input->dims.end() - 1}));
  build_steps_.push_back(
      {loc,
       [this, input, before_reshape](xnn_subgraph_t subgraph) -> absl::Status {
         MP_RETURN_IF_ERROR(MaybeAllocateIntermediateTensor(
             subgraph, before_reshape->tensor_id, *before_reshape));
         size_t reduction_axis = input->dims.size() - 1;
         RET_CHECK_EQ(
             xnn_status_success,
             xnn_define_static_mean(subgraph, 1, &reduction_axis,
                                    input->tensor_id, before_reshape->tensor_id,
                                    /*flags=*/0));
         return absl::OkStatus();
       }});

  Tensor::DimsType new_dims = input->dims;
  new_dims.back() = 1;
  return Reshape(before_reshape, std::move(new_dims));
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Rms(
    std::shared_ptr<Tensor> input, absl::SourceLocation loc) {
  ASSIGN_OR_RETURN(auto sqr_out, Square(input, loc));

  ASSIGN_OR_RETURN(auto mean_out, AvgLastDim(sqr_out, loc));

  return SquareRoot(mean_out, loc);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::RmsNorm(
    std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> scale,
    absl::SourceLocation loc) {
  ASSIGN_OR_RETURN(auto rms_out, Rms(input));

  ASSIGN_OR_RETURN(auto clamped_rms, Clamp(rms_out, {.out_min = 1e-6}));

  // div_out = input / rms
  ASSIGN_OR_RETURN(auto div_out, ElementDiv(input, clamped_rms));

  // div_out * (1 + scale) = div_out + div_out * scale
  ASSIGN_OR_RETURN(auto normed_div_out, ElementMul(div_out, scale));

  return ElementAdd(div_out, normed_div_out);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::ElementAdd(
    std::shared_ptr<Tensor> lhs, float rhs, ClampParams params,
    absl::SourceLocation loc) {
  ASSIGN_OR_RETURN(auto rhs_tensor, NewWeight({1}));
  MP_RETURN_IF_ERROR(rhs_tensor->LoadFromVec(std::vector<float>({rhs})));

  return ElementAdd(lhs, rhs_tensor, params, loc);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::ElementAdd(
    std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs,
    ClampParams params, absl::SourceLocation loc) {
  ASSIGN_OR_RETURN(auto output,
                   IntermediateTensor(OutDimsForElementwiseOp(*lhs, *rhs)));

  build_steps_.push_back(
      {loc,
       [this, lhs, rhs, output,
        params](xnn_subgraph_t subgraph) -> absl::Status {
         MP_RETURN_IF_ERROR(MaybeAllocateIntermediateTensor(subgraph, *output));
         RET_CHECK_EQ(xnn_status_success,
                      xnn_define_add2(subgraph, params.out_min, params.out_max,
                                      lhs->tensor_id, rhs->tensor_id,
                                      output->tensor_id, /*flags=*/0));
         return absl::OkStatus();
       }});

  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::ElementMul(
    std::shared_ptr<Tensor> lhs, float rhs, ClampParams params,
    absl::SourceLocation loc) {
  ASSIGN_OR_RETURN(auto rhs_tensor, NewWeight({1}));
  MP_RETURN_IF_ERROR(rhs_tensor->LoadFromVec(std::vector<float>({rhs})));

  return ElementMul(lhs, rhs_tensor, params, loc);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::ElementMul(
    std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs,
    ClampParams params, absl::SourceLocation loc) {
  ASSIGN_OR_RETURN(auto output,
                   IntermediateTensor(OutDimsForElementwiseOp(*lhs, *rhs)));

  build_steps_.push_back(
      {loc,
       [this, lhs, rhs, output,
        params](xnn_subgraph_t subgraph) -> absl::Status {
         MP_RETURN_IF_ERROR(MaybeAllocateIntermediateTensor(subgraph, *output));
         RET_CHECK_EQ(
             xnn_status_success,
             xnn_define_multiply2(subgraph, params.out_min, params.out_max,
                                  lhs->tensor_id, rhs->tensor_id,
                                  output->tensor_id, /*flags=*/0));
         return absl::OkStatus();
       }});

  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::ElementDiv(
    std::shared_ptr<Tensor> lhs, float rhs, ClampParams params,
    absl::SourceLocation loc) {
  ASSIGN_OR_RETURN(auto rhs_tensor, NewWeight({1}));
  MP_RETURN_IF_ERROR(rhs_tensor->LoadFromVec(std::vector<float>({rhs})));

  return ElementDiv(lhs, rhs_tensor, params, loc);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::ElementDiv(
    std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs,
    ClampParams params, absl::SourceLocation loc) {
  ASSIGN_OR_RETURN(auto output,
                   IntermediateTensor(OutDimsForElementwiseOp(*lhs, *rhs)));

  build_steps_.push_back(
      {loc,
       [this, lhs, rhs, output,
        params](xnn_subgraph_t subgraph) -> absl::Status {
         MP_RETURN_IF_ERROR(MaybeAllocateIntermediateTensor(subgraph, *output));
         RET_CHECK_EQ(
             xnn_status_success,
             xnn_define_divide(subgraph, params.out_min, params.out_max,
                               lhs->tensor_id, rhs->tensor_id,
                               output->tensor_id, /*flags=*/0));
         return absl::OkStatus();
       }});

  return output;
}

// TODO: write an op?
absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::PerDimScale(
    std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> per_dim_scale,
    absl::SourceLocation loc) {
  // input: B T N H
  // 1/softplus(0) = 1.442695041
  // scale = softplus(w) * 1.442695041 / np.sqrt(query.shape[-1])
  // query = query * scale
  const auto& input_dim = input->dims;
  DCHECK_GE(input_dim.size(), 1);
  const size_t H = input_dim.back();

  if (!per_dim_scale_cache_.contains(H) ||
      !per_dim_scale_cache_[H].contains(per_dim_scale.get())) {
    ASSIGN_OR_RETURN(auto cached_pds, NewWeight(per_dim_scale->dims));

    auto* pds_in = static_cast<float*>(per_dim_scale->Data());
    std::vector<float> pds_scaled(per_dim_scale->num_elements);
    SoftPlus(per_dim_scale->num_elements, input_dim, pds_in, pds_scaled.data());
    MP_RETURN_IF_ERROR(cached_pds->LoadFromVec(std::move(pds_scaled)));
    per_dim_scale_cache_[H][per_dim_scale.get()] = cached_pds;
  }

  return ElementMul(input, per_dim_scale_cache_[H][per_dim_scale.get()]);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Rope(
    std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> segment_pos,
    absl::SourceLocation loc) {
  // TODO: seg_pos should not be weight.
  rope_weigths_.insert(segment_pos);

  const auto& input_dim = input->dims;
  const auto& segment_pos_dim = segment_pos->dims;
  // B T N H
  RET_CHECK_EQ(input_dim.size(), 4) << "xnn requirement";
  // S H
  RET_CHECK_EQ(segment_pos_dim.size(), 2) << "xnn requirement";

  ASSIGN_OR_RETURN(auto output, IntermediateTensor(input_dim));

  const auto input_seq_size = input_dim[1];
  RET_CHECK_LE(input_seq_size, segment_pos_dim[0]);
  const auto head_dim_H = input_dim[3];
  RET_CHECK_EQ(head_dim_H, segment_pos_dim[1]);

  build_steps_.push_back(
      {loc,
       [this, input, output, segment_pos,
        input_seq_size](xnn_subgraph_t subgraph) -> absl::Status {
         MP_RETURN_IF_ERROR(MaybeAllocateIntermediateTensor(subgraph, *output));
         RET_CHECK_EQ(
             xnn_status_success,
             xnn_define_rope(subgraph, input_seq_size, input->tensor_id,
                             segment_pos->tensor_id, output->tensor_id,
                             /*flags=*/0));
         return absl::OkStatus();
       }});

  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::BatchMatMul(
    std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight,
    FullConnParams params, absl::SourceLocation loc) {
  const auto& lhs_dim = input->dims;
  const auto& rhs_dim = weight->dims;

  // [B, N, T, H] . [B, N, S, H], N == 12, B == 1
  DCHECK_EQ(lhs_dim.size(), 4);
  DCHECK_EQ(rhs_dim.size(), 4);
  DCHECK_EQ(lhs_dim.back(), rhs_dim.back());
  DCHECK_EQ(lhs_dim.back(), rhs_dim.back());
  constexpr size_t num_slices = 12;
  DCHECK_EQ(lhs_dim[1], num_slices);
  DCHECK_EQ(rhs_dim[1], num_slices);
  const size_t S = rhs_dim[2];
  const size_t T = lhs_dim[2];
  const size_t batch_size = lhs_dim[0] * lhs_dim[1];
  DCHECK_EQ(batch_size, rhs_dim[0] * rhs_dim[1]);
  DCHECK_EQ(batch_size, 12);

  ASSIGN_OR_RETURN(auto output, IntermediateTensor({1, 12, T, S}));

  build_steps_.push_back(
      {loc, [input, output, weight](xnn_subgraph_t subgraph) -> absl::Status {
         MP_RETURN_IF_ERROR(MaybeAllocateIntermediateTensor(subgraph, *output));

         RET_CHECK_EQ(xnn_status_success,
                      xnn_define_batch_matrix_multiply(
                          subgraph, input->tensor_id, weight->tensor_id,
                          output->tensor_id, /*flags=*/0));

         return absl::OkStatus();
       }});

  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Tanh(
    std::shared_ptr<Tensor> input, absl::SourceLocation loc) {
  ASSIGN_OR_RETURN(auto output, IntermediateTensor(input->dims));

  build_steps_.push_back(
      {loc, [this, input, output](xnn_subgraph_t subgraph) -> absl::Status {
         MP_RETURN_IF_ERROR(MaybeAllocateIntermediateTensor(subgraph, *output));

         RET_CHECK_EQ(xnn_status_success,
                      xnn_define_tanh(subgraph, input->tensor_id,
                                      output->tensor_id, /*flags=*/0));
         return absl::OkStatus();
       }});

  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::CapTanh(
    std::shared_ptr<Tensor> input, float cap, absl::SourceLocation loc) {
  ASSIGN_OR_RETURN(auto div, ElementDiv(input, cap));
  ASSIGN_OR_RETURN(auto tanh, Tanh(div));
  return ElementMul(tanh, cap);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::DotAttention(
    std::shared_ptr<Tensor> query_proj, std::shared_ptr<Tensor> key_proj,
    std::shared_ptr<Tensor> value_proj, std::shared_ptr<Tensor> atten_mask,
    std::shared_ptr<Tensor> per_dim_scale, absl::SourceLocation loc) {
  // BTNH
  ASSIGN_OR_RETURN(auto query_after_scale,
                   PerDimScale(query_proj, per_dim_scale));

  // Dot similarity
  // BTNH -> BNTH
  ASSIGN_OR_RETURN(auto query_permuted,
                   Permute(query_after_scale, {0, 2, 1, 3}));
  // BSNH -> BNSH
  ASSIGN_OR_RETURN(auto key_permuted, Permute(key_proj, {0, 2, 1, 3}));
  // einsum(BNTH.BNSH -> BNTS)
  ASSIGN_OR_RETURN(auto logits, BatchMatMul(query_permuted, key_permuted));

  // Cap, mask
  ASSIGN_OR_RETURN(auto cap_logits, CapTanh(logits, 50));
  ASSIGN_OR_RETURN(auto padded_logits, ElementAdd(atten_mask, cap_logits));
  ASSIGN_OR_RETURN(auto probs, Softmax(padded_logits));
  ASSIGN_OR_RETURN(auto value_permuted, Permute(value_proj, {0, 2, 3, 1}));

  // Outcome
  // BNTS.BNHS -> BNTH
  ASSIGN_OR_RETURN(auto outcome_before_permute,
                   BatchMatMul(probs, value_permuted));
  // [B, N, T, H] -> BTNH
  return Permute(outcome_before_permute, {0, 2, 1, 3});
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::SelfAttentionProj(
    std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight,
    absl::SourceLocation loc) {
  const auto& input_dim = input->dims;
  const auto& weight_dim = weight->dims;
  size_t N = 0, H = 0;
  RET_CHECK_EQ(input_dim.size(), 3) << "BTD";

  std::optional<int> reshaped_N =
      weight->GetMetadata(kKeySelfAttentionReshapedWeight);
  RET_CHECK(reshaped_N && *reshaped_N)
      << "We rely on " << kKeySelfAttentionReshapedWeight << " to get N";
  RET_CHECK_EQ(weight_dim.size(), 2) << "NH,D";
  N = *reshaped_N;
  H = weight_dim[0] / N;

  // out: B,T,NH
  ASSIGN_OR_RETURN(auto proj, MatMul(input, weight));

  // B,T,NH -> B,T,N,H
  return Reshape(proj, {input_dim[0], input_dim[1], N, H});
}

absl::Status XnnGraph::CreateRuntime() {
  RET_CHECK_EQ(runtime_.get(), nullptr);
  xnn_runtime_t runtime_ptr = nullptr;
  uint32_t flags = 0;
  if (runtime_configs_->xnn_profile) {
    flags |= XNN_FLAG_BASIC_PROFILING;

    if (!runtime_configs_->xnn_profile_csv.empty()) {
      MP_RETURN_IF_ERROR(file::SetContents(runtime_configs_->xnn_profile_csv,
                                           "node_id; time(us); op_name\n",
                                           file::Defaults()));
    }
  }
  pthreadpool_t threadpool =
      pthreadpool_create(runtime_configs_->xnn_num_threads);
  threadpool_ = XnnThreadpoolPtr{threadpool, pthreadpool_destroy};

  RET_CHECK_EQ(xnn_status_success,
               xnn_create_runtime_v2(owned_subgraph_.get(), threadpool, flags,
                                     &runtime_ptr));
  RET_CHECK_NE(runtime_ptr, nullptr);
  runtime_ = XnnRuntimePtr{runtime_ptr, xnn_delete_runtime};

  return absl::OkStatus();
}

absl::Status XnnGraph::SetupRuntime() {
  {
    VLOG(3) << "input size " << input_tensors_.size();
    VLOG(3) << "output size " << output_tensors_.size();
    VLOG(3) << "rope size " << rope_weigths_.size();
    externals_.clear();
    // Init external
    for (const auto& input : input_tensors_) {
      VLOG(3) << "input id " << input->tensor_id;
      externals_.push_back(xnn_external_value{input->tensor_id, input->Data()});
    }
    for (const auto& output : output_tensors_) {
      VLOG(3) << "output id " << output->tensor_id;
      externals_.push_back(
          xnn_external_value{output->tensor_id, output->Data()});
    }
    for (const auto& t : rope_weigths_) {
      VLOG(3) << "rope id " << t->tensor_id;
    }
  }
  RET_CHECK_EQ(
      xnn_status_success,
      xnn_setup_runtime(runtime_.get(), externals_.size(), externals_.data()));
  return absl::OkStatus();
}

absl::Status XnnGraph::Run() {
  RET_CHECK(runtime_);

  RET_CHECK_EQ(xnn_status_success, xnn_invoke_runtime(runtime_.get()));

  if (runtime_configs_->xnn_profile) {
    size_t required_size = 0;

    // xnn_get_runtime_profiling_info is called twice. The first time it sets
    // required_size to the required size of the buffer to store the result and
    // returns xnn_status_out_of_memory. The second time it writes the result to
    // the buffer provided that the buffer is large enough and returns
    // xnn_status_success.
    xnn_status status = xnn_get_runtime_profiling_info(
        runtime_.get(), xnn_profile_info_operator_name, /*param_value_size*/ 0,
        /*param_value*/ nullptr, &required_size);
    std::vector<char> operator_names;
    if (status == xnn_status_out_of_memory) {
      operator_names.resize(required_size);
      status = xnn_get_runtime_profiling_info(
          runtime_.get(), xnn_profile_info_operator_name, operator_names.size(),
          operator_names.data(), &required_size);
    }
    RET_CHECK_EQ(status, xnn_status_success);
    size_t num_operators;
    status = xnn_get_runtime_profiling_info(
        runtime_.get(), xnn_profile_info_num_operators, sizeof(num_operators),
        &num_operators, &required_size);
    RET_CHECK_EQ(status, xnn_status_success);
    status = xnn_get_runtime_profiling_info(
        runtime_.get(), xnn_profile_info_operator_timing,
        /*param_value_size*/ 0,
        /*param_value*/ nullptr, &required_size);
    std::vector<uint64_t> operator_timings;
    if (status == xnn_status_out_of_memory) {
      operator_timings.resize(required_size / sizeof(uint64_t));
      status = xnn_get_runtime_profiling_info(
          runtime_.get(), xnn_profile_info_operator_timing,
          operator_timings.size() * sizeof(uint64_t), operator_timings.data(),
          &required_size);
    }
    RET_CHECK_EQ(status, xnn_status_success);
    const char* operator_name = nullptr;
    size_t name_len = 0;
    std::stringstream ss;
    for (size_t node_index = 0; node_index < num_operators; ++node_index) {
      operator_name = &operator_names[name_len];
      name_len += strlen(operator_name) + 1;
      VLOG(2) << "XnnGraphBuilder::Profile() node_index: " << node_index
              << ", time: " << operator_timings[node_index] << " us, "
              << operator_name << "\n";
      if (!runtime_configs_->xnn_profile_csv.empty()) {
        // Use ';' instead of ',' because operator_name contains comma.
        ss << node_index << "; " << operator_timings[node_index] << "; "
           << operator_name << "\n";
      }
    }
    if (!runtime_configs_->xnn_profile_csv.empty()) {
      MP_RETURN_IF_ERROR(file::AppendStringToFile(
          runtime_configs_->xnn_profile_csv, ss.str(), file::Defaults()));
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Clamp(
    std::shared_ptr<Tensor> input, ClampParams params,
    absl::SourceLocation loc) {
  ASSIGN_OR_RETURN(auto output, IntermediateTensor(input->dims));

  build_steps_.push_back(
      {loc,
       [this, input, output, params](xnn_subgraph_t subgraph) -> absl::Status {
         MP_RETURN_IF_ERROR(MaybeAllocateIntermediateTensor(subgraph, *output));

         RET_CHECK_EQ(xnn_status_success,
                      xnn_define_clamp(subgraph, params.out_min, params.out_max,
                                       input->tensor_id, output->tensor_id,
                                       /*flags=*/0));
         return absl::OkStatus();
       }});

  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Gelu(
    std::shared_ptr<Tensor> input, absl::SourceLocation loc) {
  // x^2
  ASSIGN_OR_RETURN(auto sqr_out, Square(input));

  // 0.044715 * x^2
  ASSIGN_OR_RETURN(auto sqr_4471, ElementMul(sqr_out, 0.044715));

  // 1 + 0.044715 * x^2
  ASSIGN_OR_RETURN(auto sqr_4471_1, ElementAdd(sqr_4471, 1.0f));

  // x + 0.044715 * x^3
  ASSIGN_OR_RETURN(auto x_cube_4471, ElementMul(sqr_4471_1, input));

  constexpr float sqrt_2_over_pi = 0.7978845608;
  ASSIGN_OR_RETURN(auto sqrt_2_over_pi_x_cube_4471,
                   ElementMul(x_cube_4471, sqrt_2_over_pi));

  // tanh(x + 0.044715 * x^3)
  ASSIGN_OR_RETURN(auto tanh_x_cube_4471, Tanh(sqrt_2_over_pi_x_cube_4471));

  // 1 + tanh(x + 0.044715 * x^3)
  ASSIGN_OR_RETURN(auto tanh_x_cube_4471_1, ElementAdd(tanh_x_cube_4471, 1.0f));

  // 0.5 * (1 + [tanh(x + 0.044715 * x^3)])
  ASSIGN_OR_RETURN(auto cdf, ElementMul(tanh_x_cube_4471_1, 0.5));

  return ElementMul(input, cdf);
}

}  // namespace xnn_utils
}  // namespace mediapipe

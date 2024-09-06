// Copyright 2024 The MediaPipe Authors.
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

#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/graph_builder.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/utils.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/xnn_tensor.h"
#include "xnnpack.h"  // from @XNNPACK

namespace mediapipe::tasks::genai {
namespace xnn_utils {
namespace {

absl::Status AppendStringToFile(absl::string_view file,
                                absl::string_view contents) {
  std::ofstream ofstr(std::string(file), std::ios::app);
  RET_CHECK(ofstr);
  ofstr << contents;
  return absl::OkStatus();
}

// XNNPACK supports broadcasting, this function inferences the output shape
// based on input tensor shapes.
std::vector<size_t> OutDimsForElementwiseOp(const Tensor& lhs,
                                            const Tensor& rhs) {
  ABSL_DCHECK(!lhs.dims.empty());
  ABSL_DCHECK(!rhs.dims.empty());
  std::vector<size_t> lhs_dims_rev(lhs.dims.rbegin(), lhs.dims.rend());
  std::vector<size_t> rhs_dims_rev(rhs.dims.rbegin(), rhs.dims.rend());
  ABSL_DCHECK([&]() -> bool {
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

// 1.0/softplus(0.0) = 1.442695041
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

XnnWeightsCache::XnnWeightsCache(xnn_weights_cache_t weights_cache)
    : xnn_weights_cache(weights_cache) {}

XnnWeightsCache::~XnnWeightsCache() {
  xnn_delete_weights_cache(xnn_weights_cache);
}

absl::Status XnnWeightsCache::Finalize() {
  RET_CHECK_NE(Get(), nullptr);
  RET_CHECK_EQ(xnn_status_success,
               xnn_finalize_weights_cache(
                   Get(), xnn_weights_cache_finalization_kind_hard));
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<XnnWeightsCache>> CreateWeightsCache(
    size_t buffer_size) {
  RET_CHECK_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_weights_cache_t weights_cache = nullptr;
  RET_CHECK_EQ(xnn_status_success,
               xnn_create_weights_cache_with_size(buffer_size, &weights_cache));
  return std::make_shared<XnnWeightsCache>(weights_cache);
}

absl::StatusOr<std::unique_ptr<XnnGraph>> XnnGraphBuilder::Build() {
  VLOG(2) << "XnnGraphBuilder::Build() building...";
  RET_CHECK_EQ(xnn_status_success, xnn_initialize(nullptr));

  std::vector<std::shared_ptr<Tensor>> output_tensors;
  {
    for (auto& t : interm_tensors_added_order_) {
      if (interm_tensors_.contains(t) && t->is_output_tensor) {
        output_tensors.push_back(t);
      }
    }
    for (auto& t : output_tensors) {
      interm_tensors_.erase(t);
    }
  }

  xnn_subgraph_t subgraph_ptr = nullptr;
  RET_CHECK_EQ(xnn_status_success,
               xnn_create_subgraph(
                   /*external_value_ids=*/input_tensors_added_order_.size() +
                       output_tensors.size(),
                   /*flags=*/0, &subgraph_ptr));
  RET_CHECK_NE(subgraph_ptr, nullptr);

  {
    uint32_t cnt = 0;
    for (auto& t : input_tensors_added_order_) {
      t->set_tensor_id(subgraph_ptr, cnt++);
    }
    for (auto& t : output_tensors) {
      RET_CHECK_EQ(t->tensor_id(subgraph_ptr), XNN_INVALID_VALUE_ID);
      t->set_tensor_id(subgraph_ptr, cnt++);
    }
  }

  XnnSubgraphPtr subgraph{subgraph_ptr, xnn_delete_subgraph};

  for (auto& t : static_weights_) {
    MP_RETURN_IF_ERROR(t->DefineWeight(*subgraph_ptr));
  }
  for (auto& input : input_tensors_added_order_) {
    MP_RETURN_IF_ERROR(input->DefineAsInput(*subgraph));
  }
  for (auto& output : output_tensors) {
    MP_RETURN_IF_ERROR(output->DefineAsOutput(*subgraph));
  }

  for (auto& step : build_steps_) {
    if (auto s = step(subgraph.get()); !s.ok()) {
      return s;
    }
  }

  build_steps_.clear();
  XnnGraph result(std::move(subgraph),
                  std::make_unique<RuntimeConfigs>(*runtime_configs_));
  result.input_tensors_ = std::move(input_tensors_added_order_);
  result.output_tensors_ = std::move(output_tensors);
  result.static_weights_ = std::move(static_weights_);

  VLOG(2) << "XnnGraphBuilder::Build() creating runtime...";
  MP_RETURN_IF_ERROR(result.CreateRuntime());
  if (!result.runtime_configs_->weights_cache) {
    VLOG(2) << "XnnGraphBuilder::Build() setting up runtime...";
    MP_RETURN_IF_ERROR(result.SetupRuntime());
  }
  VLOG(2) << "XnnGraphBuilder::Build() done";
  return std::make_unique<XnnGraph>(std::move(result));
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::NewInput(
    Tensor::DimsType dims, absl::string_view tag) {
  auto t = std::make_shared<Tensor>(std::move(dims), data_type_);
  t->AllocateBufferIfNeeded();
  t->tag = tag;
  MP_RETURN_IF_ERROR(MarkInput(t));
  return t;
}

absl::Status XnnGraphBuilder::MarkInput(std::shared_ptr<Tensor> t) {
  input_tensors_.insert(t);
  input_tensors_added_order_.push_back(t);
  return absl::OkStatus();
}

void XnnGraphBuilder::NewWeight(std::shared_ptr<Tensor> t) {
  if (interm_tensors_.contains(t) || input_tensors_.contains(t)) {
    return;
  }

  static_weights_.insert(t);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::IntermediateTensor(
    Tensor::DimsType dims, absl::string_view tag) {
  return IntermediateTensor(dims, data_type_, tag);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::IntermediateTensor(
    Tensor::DimsType dims, xnn_datatype data_type, absl::string_view tag) {
  auto t = std::make_shared<Tensor>(std::move(dims), data_type);
  t->tag = tag;

  build_steps_.push_back([this, t](xnn_subgraph_t subgraph) -> absl::Status {
    // Could be moved to output tensors, thus need check.
    if (interm_tensors_.contains(t)) {
      return t->DefineAsIntermediateTensor(*subgraph);
    }
    return absl::OkStatus();
  });

  interm_tensors_.insert(t);
  interm_tensors_added_order_.push_back(t);
  return t;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Reshape(
    std::shared_ptr<Tensor> input, Tensor::DimsType new_dims) {
  size_t output_axis_dynamic = new_dims.size();
  Tensor::DimsType output_dims = new_dims;

  // Compute output shape.
  for (size_t dim_idx = 0; dim_idx < new_dims.size(); ++dim_idx) {
    if (new_dims[dim_idx] == 0) {
      if (output_axis_dynamic < new_dims.size()) {
        return absl::InvalidArgumentError(
            absl::StrCat("More than one dynamic dimension: ",
                         absl::StrJoin(new_dims, ", ")));
      }
      output_axis_dynamic = dim_idx;
      output_dims[dim_idx] = 1;
    }
  }
  if (output_axis_dynamic < new_dims.size()) {
    const size_t input_num_elements =
        std::accumulate(std::begin(input->dims), std::end(input->dims),
                        size_t(1), std::multiplies<size_t>());
    const size_t output_num_elements =
        std::accumulate(std::begin(output_dims), std::end(output_dims),
                        size_t(1), std::multiplies<size_t>());
    const size_t inferred_dim = input_num_elements / output_num_elements;
    if (inferred_dim * output_num_elements != input_num_elements) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Cannot properly infer input [", absl::StrJoin(input->dims, ", "),
          "] given hint [", absl::StrJoin(new_dims, ", "), "]"));
    }
    output_dims[output_axis_dynamic] = inferred_dim;
  }

  MP_ASSIGN_OR_RETURN(auto output, IntermediateTensor(std::move(output_dims),
                                                      "reshape_output"));
  RET_CHECK_EQ(input->num_elements, output->num_elements)
      << "otherwise reshape does not make sense. input dimension "
      << input->dims << " output dimension " << output->dims;

  build_steps_.push_back([input, output,
                          new_dims](xnn_subgraph_t subgraph) -> absl::Status {
    RET_CHECK_EQ(
        xnn_status_success,
        xnn_define_static_reshape(subgraph, new_dims.size(), new_dims.data(),
                                  input->tensor_id(subgraph),
                                  output->tensor_id(subgraph), /*flags=*/0));
    return absl::OkStatus();
  });
  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::FullConn(
    std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight,
    std::shared_ptr<Tensor> bias, FullConnParams params) {
  RET_CHECK(weight);
  const auto& input_dim = input->dims;
  const auto& weight_dim = weight->dims;
  ABSL_DCHECK_GT(input_dim.size(), 1);
  ABSL_DCHECK_GE(weight_dim.size(), 2);
  if (weight_dim.size() == 3) {
    RET_CHECK_EQ(weight_dim[0], 1);
  } else if (weight_dim.size() == 4) {
    RET_CHECK_EQ(weight_dim[0], 1);
    RET_CHECK_EQ(weight_dim[1], 1);
  }
  NewWeight(weight);

  if (bias) {
    RET_CHECK_LE(bias->dims.size(), 1);
    NewWeight(bias);
  }

  Tensor::DimsType out_dims = input_dim;
  // Not considering reshape 2D
  if (params.transpose) {
    RET_CHECK_EQ(weight_dim.size(), 2) << "otherwise change following line";
    RET_CHECK_EQ(input_dim.back(), *(weight_dim.end() - 2)) << *weight;
    out_dims.back() = weight_dim.back();
  } else {
    RET_CHECK_EQ(input_dim.back(), weight_dim.back()) << *weight;
    out_dims.pop_back();
    for (size_t i = 0; i < weight_dim.size() - 1; ++i) {
      // NHD . BTD -> NHBT
      out_dims.push_back(weight_dim[i]);
    }
  }

  std::shared_ptr<Tensor> qd_input;
  bool use_dynamic_quantization = false;
  if (runtime_configs_->use_dynamic_quantization.has_value()) {
    use_dynamic_quantization =
        runtime_configs_->use_dynamic_quantization.value();
  } else if (weight->datatype == xnn_datatype_qcint8 ||
             weight->datatype == xnn_datatype_qcint4) {
    use_dynamic_quantization = true;
  }
  VLOG(3) << "use_dynamic_quantization: " << use_dynamic_quantization;
  if (use_dynamic_quantization) {
    MP_ASSIGN_OR_RETURN(
        qd_input, IntermediateTensor({input->dims.begin(), input->dims.end()},
                                     xnn_datatype_qdint8));
  }

  // TODO: b/295116789 - work around.
  if (!use_dynamic_quantization &&
      (input->datatype == xnn_datatype_fp32 &&
       (weight->datatype == xnn_datatype_qcint8 ||
        weight->datatype == xnn_datatype_qcint4)) &&
      bias) {
    constexpr absl::string_view kKey = "295116789_workaround";
    auto workaround_applied = bias->GetMetadata(kKey);
    if (!workaround_applied || !*workaround_applied) {
      auto* qc_weight = static_cast<QCTensor*>(weight.get());
      RET_CHECK_EQ(bias->num_elements, weight_dim[qc_weight->dim_scale]);
      float* bias_array = bias->DataAs<float>();
      float* scale_array = qc_weight->scale_data.get();
      std::vector<float> workaround_bias;
      for (size_t i = 0; i < bias->num_elements; ++i) {
        workaround_bias.push_back(bias_array[i] / scale_array[i]);
      }
      MP_RETURN_IF_ERROR(bias->LoadFromVec(std::move(workaround_bias)));
      bias->SetMetadata(kKey, 1);
    }
  }

  MP_ASSIGN_OR_RETURN(
      auto output, IntermediateTensor(std::move(out_dims), "full_conn_output"));

  build_steps_.push_back([input, weight, bias, params, output,
                          qd_input](xnn_subgraph_t subgraph) -> absl::Status {
    if (qd_input) {
      // Set XNN_FLAG_MAYBE_PACK_FOR_GEMM if the weights are 4 bit.
      uint32_t flags = weight->datatype == xnn_datatype_qcint4 ? 0x00000080 : 0;
      RET_CHECK_EQ(xnn_status_success,
                   xnn_define_convert(subgraph, input->tensor_id(subgraph),
                                      qd_input->tensor_id(subgraph), flags));
      RET_CHECK_EQ(
          xnn_status_success,
          xnn_define_fully_connected(
              subgraph, params.out_min, params.out_max,
              qd_input->tensor_id(subgraph), weight->tensor_id(subgraph),
              bias ? bias->tensor_id(subgraph) : XNN_INVALID_VALUE_ID,
              output->tensor_id(subgraph),
              /*flags=*/params.transpose ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0));
    } else {
      RET_CHECK_EQ(
          xnn_status_success,
          xnn_define_fully_connected(
              subgraph, params.out_min, params.out_max,
              input->tensor_id(subgraph), weight->tensor_id(subgraph),
              bias ? bias->tensor_id(subgraph) : XNN_INVALID_VALUE_ID,
              output->tensor_id(subgraph),
              /*flags=*/params.transpose ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0));
    }

    return absl::OkStatus();
  });
  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Permute(
    std::shared_ptr<Tensor> input, Tensor::DimsType permute) {
  RET_CHECK_EQ(input->dims.size(), permute.size());
  const auto& old_dims = input->dims;
  std::vector<size_t> new_dims;
  for (size_t i = 0; i < permute.size(); ++i) {
    new_dims.push_back(old_dims[permute[i]]);
  }
  MP_ASSIGN_OR_RETURN(
      auto output, IntermediateTensor(std::move(new_dims), "permute_output"));

  build_steps_.push_back([permute, input,
                          output](xnn_subgraph_t subgraph) -> absl::Status {
    RET_CHECK_EQ(
        xnn_status_success,
        xnn_define_static_transpose(subgraph, permute.size(), permute.data(),
                                    input->tensor_id(subgraph),
                                    output->tensor_id(subgraph), /*flags=*/0));
    return absl::OkStatus();
  });
  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Slice(
    std::shared_ptr<Tensor> input, Tensor::DimsType starts,
    Tensor::DimsType ends) {
  const auto& input_dims = input->dims;
  RET_CHECK_EQ(input_dims.size(), starts.size());
  RET_CHECK_EQ(input_dims.size(), ends.size());
  Tensor::DimsType sizes;
  sizes.reserve(input_dims.size());
  for (size_t i = 0; i < starts.size(); ++i) {
    RET_CHECK_LT(starts[i], ends[i]);
    RET_CHECK_LE(ends[i], input_dims[i]);
    size_t size = ends[i] - starts[i];
    RET_CHECK_GT(size, 0);
    sizes.push_back(size);
  }
  MP_ASSIGN_OR_RETURN(auto output, IntermediateTensor(sizes, "slice_output"));

  build_steps_.push_back(
      [input, output, starts, sizes](xnn_subgraph_t subgraph) -> absl::Status {
        RET_CHECK_EQ(
            xnn_status_success,
            xnn_define_static_slice(subgraph, starts.size(), starts.data(),
                                    sizes.data(), input->tensor_id(subgraph),
                                    output->tensor_id(subgraph), /*flags=*/0));
        return absl::OkStatus();
      });
  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Slice(
    std::shared_ptr<Tensor> input, size_t axis, size_t offset, size_t length) {
  const auto& input_dims = input->dims;
  Tensor::DimsType offsets(input_dims.size(), 0);
  offsets[axis] = offset;
  Tensor::DimsType output_dims = input_dims;
  output_dims[axis] = length;
  Tensor::DimsType inferrable_output_dims(input_dims.size(), 0);
  inferrable_output_dims[axis] = length;

  MP_ASSIGN_OR_RETURN(auto output,
                      IntermediateTensor(output_dims, "slice_output"));

  build_steps_.push_back([input, output, offsets, inferrable_output_dims](
                             xnn_subgraph_t subgraph) -> absl::Status {
    RET_CHECK_EQ(xnn_status_success,
                 xnn_define_static_slice(
                     subgraph, offsets.size(), offsets.data(),
                     inferrable_output_dims.data(), input->tensor_id(subgraph),
                     output->tensor_id(subgraph), /*flags=*/0));
    return absl::OkStatus();
  });
  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Concat(
    size_t axis, std::shared_ptr<Tensor> input1,
    std::shared_ptr<Tensor> input2) {
  RET_CHECK_EQ(input1->dims.size(), input2->dims.size());
  RET_CHECK_LT(axis, input1->dims.size());
  Tensor::DimsType output_dims;
  for (int i = 0; i < input1->dims.size(); ++i) {
    if (i == axis) {
      output_dims.push_back(input1->dims[i] + input2->dims[i]);
    } else {
      RET_CHECK_EQ(input1->dims[i], input2->dims[i]);
      output_dims.push_back(input1->dims[i]);
    }
  }
  MP_ASSIGN_OR_RETURN(auto output,
                      IntermediateTensor(output_dims, "concat_output"));

  build_steps_.push_back(
      [axis, input1, input2, output](xnn_subgraph_t subgraph) -> absl::Status {
        RET_CHECK_EQ(
            xnn_status_success,
            xnn_define_concatenate2(subgraph, axis, input1->tensor_id(subgraph),
                                    input2->tensor_id(subgraph),
                                    output->tensor_id(subgraph), /*flags=*/0));
        return absl::OkStatus();
      });
  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Square(
    std::shared_ptr<Tensor> input) {
  MP_ASSIGN_OR_RETURN(auto output,
                      IntermediateTensor(input->dims, "square_output"));

  build_steps_.push_back(
      [output, input](xnn_subgraph_t subgraph) -> absl::Status {
        RET_CHECK_EQ(xnn_status_success,
                     xnn_define_square(subgraph, input->tensor_id(subgraph),
                                       output->tensor_id(subgraph),
                                       /*flags=*/0));
        return absl::Status();
      });

  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Softmax(
    std::shared_ptr<Tensor> input) {
  MP_ASSIGN_OR_RETURN(auto output,
                      IntermediateTensor(input->dims, "softmax_output"));

  build_steps_.push_back(
      [output, input](xnn_subgraph_t subgraph) -> absl::Status {
        RET_CHECK_EQ(xnn_status_success,
                     xnn_define_softmax(subgraph, input->tensor_id(subgraph),
                                        output->tensor_id(subgraph),
                                        /*flags=*/0));
        return absl::Status();
      });

  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::SquareRoot(
    std::shared_ptr<Tensor> input) {
  MP_ASSIGN_OR_RETURN(auto output,
                      IntermediateTensor(input->dims, "square_root_output"));

  build_steps_.push_back([output,
                          input](xnn_subgraph_t subgraph) -> absl::Status {
    RET_CHECK_EQ(xnn_status_success,
                 xnn_define_square_root(subgraph, input->tensor_id(subgraph),
                                        output->tensor_id(subgraph),
                                        /*flags=*/0));
    return absl::Status();
  });

  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::AvgLastDim(
    std::shared_ptr<Tensor> input) {
  Tensor::DimsType output_dims(input->dims);
  output_dims.back() = 1;
  MP_ASSIGN_OR_RETURN(auto output, IntermediateTensor(std::move(output_dims),
                                                      "avg_last_dim_output"));
  // TODO: b/149120844 - Remove the following messy code once we have settled
  // which flag to use.
#if defined(XNN_FLAG_KEEP_DIMS)
  build_steps_.push_back(
      [input, output](xnn_subgraph_t subgraph) -> absl::Status {
        size_t reduction_axis = input->dims.size() - 1;
        RET_CHECK_EQ(xnn_status_success,
                     xnn_define_static_mean(subgraph, 1, &reduction_axis,
                                            input->tensor_id(subgraph),
                                            output->tensor_id(subgraph),
                                            /*flags=*/XNN_FLAG_KEEP_DIMS));
        return absl::OkStatus();
      });
#elif defined(XNN_FLAG_REDUCE_DIMS)
  build_steps_.push_back(
      [input, output](xnn_subgraph_t subgraph) -> absl::Status {
        size_t reduction_axis = input->dims.size() - 1;
        RET_CHECK_EQ(xnn_status_success,
                     xnn_define_static_mean(subgraph, 1, &reduction_axis,
                                            input->tensor_id(subgraph),
                                            output->tensor_id(subgraph),
                                            /*flags=*/0));
        return absl::OkStatus();
      });
#else
#error one of the above flag should be defined.
#endif  // XNN_FLAG_KEEP_DIMS

  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Rms(
    std::shared_ptr<Tensor> input) {
  MP_ASSIGN_OR_RETURN(auto sqr_out, Square(input));

  MP_ASSIGN_OR_RETURN(auto mean_out, AvgLastDim(sqr_out));

  return SquareRoot(mean_out);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::RmsNorm(
    std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> scale) {
  MP_ASSIGN_OR_RETURN(auto rms_out, Rms(input));

  MP_ASSIGN_OR_RETURN(auto clamped_rms, Clamp(rms_out, {.out_min = 1e-6}));

  // div_out = input / rms
  MP_ASSIGN_OR_RETURN(auto div_out, ElementDiv(input, clamped_rms));

  // div_out * (1 + scale) = div_out + div_out * scale
  MP_ASSIGN_OR_RETURN(auto normed_div_out, ElementMul(div_out, scale));

  return ElementAdd(div_out, normed_div_out);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::ElementAdd(
    std::shared_ptr<Tensor> lhs, float rhs, ClampParams params) {
  auto rhs_tensor =
      std::make_shared<Tensor>(Tensor::DimsType{1}, xnn_datatype_fp32);
  MP_RETURN_IF_ERROR(rhs_tensor->LoadFromVec(std::vector<float>({rhs})));

  return ElementAdd(lhs, rhs_tensor, params);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::ElementAdd(
    std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs,
    ClampParams params) {
  MP_ASSIGN_OR_RETURN(auto output,
                      IntermediateTensor(OutDimsForElementwiseOp(*lhs, *rhs),
                                         "element_add_output"));
  NewWeight(rhs);

  build_steps_.push_back(
      [lhs, rhs, output, params](xnn_subgraph_t subgraph) -> absl::Status {
        RET_CHECK_EQ(
            xnn_status_success,
            xnn_define_add2(subgraph, params.out_min, params.out_max,
                            lhs->tensor_id(subgraph), rhs->tensor_id(subgraph),
                            output->tensor_id(subgraph), /*flags=*/0));
        return absl::OkStatus();
      });

  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::ElementSub(
    float lhs, std::shared_ptr<Tensor> rhs, ClampParams params) {
  auto lhs_tensor =
      std::make_shared<Tensor>(Tensor::DimsType{1}, xnn_datatype_fp32);
  MP_RETURN_IF_ERROR(lhs_tensor->LoadFromVec(std::vector<float>({lhs})));

  return ElementSub(lhs_tensor, rhs, params);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::ElementSub(
    std::shared_ptr<Tensor> lhs, float rhs, ClampParams params) {
  auto rhs_tensor =
      std::make_shared<Tensor>(Tensor::DimsType{1}, xnn_datatype_fp32);
  MP_RETURN_IF_ERROR(rhs_tensor->LoadFromVec(std::vector<float>({rhs})));

  return ElementSub(lhs, rhs_tensor, params);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::ElementSub(
    std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs,
    ClampParams params) {
  MP_ASSIGN_OR_RETURN(auto output,
                      IntermediateTensor(OutDimsForElementwiseOp(*lhs, *rhs),
                                         "element_sub_output"));
  NewWeight(lhs);
  NewWeight(rhs);

  build_steps_.push_back([lhs, rhs, output,
                          params](xnn_subgraph_t subgraph) -> absl::Status {
    RET_CHECK_EQ(
        xnn_status_success,
        xnn_define_subtract(subgraph, params.out_min, params.out_max,
                            lhs->tensor_id(subgraph), rhs->tensor_id(subgraph),
                            output->tensor_id(subgraph), /*flags=*/0));
    return absl::OkStatus();
  });

  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::ElementMul(
    std::shared_ptr<Tensor> lhs, float rhs, ClampParams params) {
  auto rhs_tensor =
      std::make_shared<Tensor>(Tensor::DimsType{1}, xnn_datatype_fp32);
  MP_RETURN_IF_ERROR(rhs_tensor->LoadFromVec(std::vector<float>({rhs})));

  return ElementMul(lhs, rhs_tensor, params);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::ElementMul(
    std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs,
    ClampParams params) {
  MP_ASSIGN_OR_RETURN(auto output,
                      IntermediateTensor(OutDimsForElementwiseOp(*lhs, *rhs),
                                         "element_mul_output"));
  NewWeight(rhs);

  build_steps_.push_back([lhs, rhs, output,
                          params](xnn_subgraph_t subgraph) -> absl::Status {
    RET_CHECK_EQ(
        xnn_status_success,
        xnn_define_multiply2(subgraph, params.out_min, params.out_max,
                             lhs->tensor_id(subgraph), rhs->tensor_id(subgraph),
                             output->tensor_id(subgraph), /*flags=*/0));
    return absl::OkStatus();
  });

  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::ElementDiv(
    std::shared_ptr<Tensor> lhs, float rhs, ClampParams params) {
  auto rhs_tensor =
      std::make_shared<Tensor>(Tensor::DimsType{1}, xnn_datatype_fp32);
  MP_RETURN_IF_ERROR(rhs_tensor->LoadFromVec(std::vector<float>({rhs})));
  NewWeight(rhs_tensor);

  return ElementDiv(lhs, rhs_tensor, params);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::ElementDiv(
    std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs,
    ClampParams params) {
  MP_ASSIGN_OR_RETURN(auto output,
                      IntermediateTensor(OutDimsForElementwiseOp(*lhs, *rhs),
                                         "element_div_output"));

  build_steps_.push_back([lhs, rhs, output,
                          params](xnn_subgraph_t subgraph) -> absl::Status {
    RET_CHECK_EQ(
        xnn_status_success,
        xnn_define_divide(subgraph, params.out_min, params.out_max,
                          lhs->tensor_id(subgraph), rhs->tensor_id(subgraph),
                          output->tensor_id(subgraph), /*flags=*/0));
    return absl::OkStatus();
  });

  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::PerDimScale(
    std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> per_dim_scale) {
  // input: B T N H
  // 1/softplus(0) = 1.442695041
  // scale = softplus(w) * 1.442695041 / np.sqrt(query.shape[-1])
  // query = query * scale
  const auto& input_dim = input->dims;
  ABSL_DCHECK_GE(input_dim.size(), 1);
  const size_t H = input_dim.back();

  if (!per_dim_scale_cache_.contains(H) ||
      !per_dim_scale_cache_[H].contains(per_dim_scale.get())) {
    auto cached_pds =
        std::make_shared<Tensor>(per_dim_scale->dims, xnn_datatype_fp32);
    NewWeight(cached_pds);

    auto* pds_in = static_cast<float*>(per_dim_scale->Data());
    std::vector<float> pds_scaled(per_dim_scale->num_elements);
    SoftPlus(per_dim_scale->num_elements, input_dim, pds_in, pds_scaled.data());
    MP_RETURN_IF_ERROR(cached_pds->LoadFromVec(std::move(pds_scaled)));
    per_dim_scale_cache_[H][per_dim_scale.get()] = cached_pds;
  }

  return ElementMul(input, per_dim_scale_cache_[H][per_dim_scale.get()]);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Rope(
    std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> segment_pos) {
  const auto& input_dim = input->dims;
  const auto& segment_pos_dim = segment_pos->dims;
  // B T N H
  RET_CHECK_EQ(input_dim.size(), 4) << "xnn requirement";
  // S H
  RET_CHECK_EQ(segment_pos_dim.size(), 2) << "xnn requirement";

  MP_ASSIGN_OR_RETURN(auto output,
                      IntermediateTensor(input_dim, "rope_output"));

  const auto input_seq_size = input_dim[1];
  RET_CHECK_LE(input_seq_size, segment_pos_dim[0]);
  const auto head_dim_H = input_dim[3];
  RET_CHECK_EQ(head_dim_H, segment_pos_dim[1]);

  build_steps_.push_back([input, output, segment_pos, input_seq_size](
                             xnn_subgraph_t subgraph) -> absl::Status {
    RET_CHECK_EQ(
        xnn_status_success,
        xnn_define_rope(subgraph, input_seq_size, input->tensor_id(subgraph),
                        segment_pos->tensor_id(subgraph),
                        output->tensor_id(subgraph),
                        /*flags=*/0));
    return absl::OkStatus();
  });

  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::PartialRope(
    std::shared_ptr<Tensor> input, size_t idx,
    std::shared_ptr<Tensor> segment_pos) {
  // B,T,N,H (Slicing along H)
  RET_CHECK_EQ(input->dims.size(), 4);

  MP_ASSIGN_OR_RETURN(auto rope_slice,
                      Slice(input, /*axis=*/3, /*offset=*/0, /*length=*/idx));
  MP_ASSIGN_OR_RETURN(auto pass_slice,
                      Slice(input, /*axis=*/3, /*offset=*/idx,
                            /*length=*/input->dims.back() - idx));
  MP_ASSIGN_OR_RETURN(auto rope, Rope(rope_slice, segment_pos));

  return Concat(/*axis=*/3, rope, pass_slice);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::BatchMatMul(
    std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight,
    FullConnParams params) {
  const auto& lhs_dim = input->dims;
  const auto& rhs_dim = weight->dims;

  // [B, N, T, S] . [B, N', H, S]
  RET_CHECK_EQ(lhs_dim.size(), 4);
  RET_CHECK_EQ(rhs_dim.size(), 4);
  uint32_t flags = 0;
  const size_t N = std::max(lhs_dim[1], rhs_dim[1]);
  const size_t T = lhs_dim[2];
  size_t H;
  if (params.transpose) {
    RET_CHECK_EQ(lhs_dim.back(), rhs_dim.back());
    flags = XNN_FLAG_TRANSPOSE_B;
    H = rhs_dim[2];
  } else {
    RET_CHECK_EQ(lhs_dim.back(), rhs_dim[rhs_dim.size() - 2]);
    H = rhs_dim[3];
  }

  NewWeight(weight);
  MP_ASSIGN_OR_RETURN(auto output, IntermediateTensor({lhs_dim[0], N, T, H},
                                                      "batch_mat_mul_output"));

  build_steps_.push_back([input, output, weight,
                          flags](xnn_subgraph_t subgraph) -> absl::Status {
    RET_CHECK_EQ(xnn_status_success, xnn_define_batch_matrix_multiply(
                                         subgraph, input->tensor_id(subgraph),
                                         weight->tensor_id(subgraph),
                                         output->tensor_id(subgraph), flags));

    return absl::OkStatus();
  });

  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Tanh(
    std::shared_ptr<Tensor> input) {
  MP_ASSIGN_OR_RETURN(auto output,
                      IntermediateTensor(input->dims, "tanh_output"));

  build_steps_.push_back(
      [input, output](xnn_subgraph_t subgraph) -> absl::Status {
        RET_CHECK_EQ(xnn_status_success,
                     xnn_define_tanh(subgraph, input->tensor_id(subgraph),
                                     output->tensor_id(subgraph), /*flags=*/0));
        return absl::OkStatus();
      });

  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::CapTanh(
    std::shared_ptr<Tensor> input, float cap) {
  RET_CHECK_GT(cap, 0.0f);

  MP_ASSIGN_OR_RETURN(auto div, ElementDiv(input, cap));
  MP_ASSIGN_OR_RETURN(auto tanh, Tanh(div));
  return ElementMul(tanh, cap);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::SquaredDifference(
    std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs) {
  MP_ASSIGN_OR_RETURN(
      auto output, IntermediateTensor(lhs->dims, "squared_difference_output"));

  build_steps_.push_back(
      [lhs, rhs, output](xnn_subgraph_t subgraph) -> absl::Status {
        RET_CHECK_EQ(xnn_status_success,
                     xnn_define_squared_difference(
                         subgraph, lhs->tensor_id(subgraph),
                         rhs->tensor_id(subgraph), output->tensor_id(subgraph),
                         /*flags=*/0));
        return absl::OkStatus();
      });

  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::LayerNorm(
    std::shared_ptr<Tensor> input, float epsilon, std::shared_ptr<Tensor> gamma,
    std::shared_ptr<Tensor> beta) {
  // This implementation is intended for text data which is usually formatted
  // as B,T,NH and normalized along the last axis.
  RET_CHECK_EQ(input->dims.size(), 3);
  MP_ASSIGN_OR_RETURN(auto mean, AvgLastDim(input));
  MP_ASSIGN_OR_RETURN(auto diff, ElementSub(input, mean));
  MP_ASSIGN_OR_RETURN(auto sq_diff, Square(diff));
  MP_ASSIGN_OR_RETURN(auto var, AvgLastDim(sq_diff));
  MP_ASSIGN_OR_RETURN(auto perturbed_var, ElementAdd(var, epsilon));
  MP_ASSIGN_OR_RETURN(auto standard_dev, SquareRoot(perturbed_var));
  MP_ASSIGN_OR_RETURN(auto normalized, ElementDiv(diff, standard_dev));
  if (gamma != nullptr) {
    RET_CHECK_EQ(gamma->dims.size(), input->dims.size());
    RET_CHECK_EQ(gamma->dims[2], input->dims[2]);
    MP_ASSIGN_OR_RETURN(normalized, ElementMul(normalized, gamma));
  }
  if (beta != nullptr) {
    RET_CHECK_EQ(beta->dims.size(), input->dims.size());
    RET_CHECK_EQ(beta->dims[2], input->dims[2]);
    MP_ASSIGN_OR_RETURN(normalized, ElementAdd(normalized, beta));
  }
  return normalized;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::SelfAttentionProj(
    std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight,
    std::shared_ptr<Tensor> bias, size_t num_heads) {
  const auto& input_dim = input->dims;
  RET_CHECK_EQ(input_dim.size(), 3) << "BTD";
  const auto& weight_dim = weight->dims;
  RET_CHECK_EQ(weight_dim.size(), 2) << "H,D or NH,D";
  const size_t B = input_dim[0];
  const size_t H = weight_dim[0] / num_heads;

  // Key exists -> [NH,D] -> no transpose.
  FullConnParams params;
  params.transpose = !weight->GetMetadata(kKeyInDimLastInWeight, 0);

  // out: B,T,NH
  MP_ASSIGN_OR_RETURN(auto proj, FullConn(input, weight, bias, params));
  // B,T,NH -> B,T,N,H
  return Reshape(proj, {B, 0, num_heads, H});
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::SelfAttentionProj(
    std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight,
    std::shared_ptr<Tensor> bias) {
  std::optional<int> reshaped_N =
      weight->GetMetadata(kKeySelfAttentionReshapedWeight);
  RET_CHECK(reshaped_N && *reshaped_N)
      << "We rely on " << kKeySelfAttentionReshapedWeight << " to get N";
  return SelfAttentionProj(input, weight, bias, *reshaped_N);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::SelfAttentionProj(
    std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> weight) {
  return SelfAttentionProj(input, weight, /*bias=*/nullptr);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::QKVAttention(
    std::shared_ptr<Tensor> query, std::shared_ptr<Tensor> key_or_value,
    Tensor::DimsType reshape_hint) {
  RET_CHECK_EQ(query->dims.size(), 4);
  RET_CHECK_EQ(key_or_value->dims.size(), 4);
  FullConnParams params{.transpose = true};
  return BatchMatMul(query, key_or_value, params);
}

absl::Status XnnGraph::CreateRuntime() {
  RET_CHECK_EQ(runtime_.get(), nullptr);
  RET_CHECK(owned_subgraph_);
  xnn_runtime_t runtime_ptr = nullptr;
  uint32_t flags = 0;
  if (runtime_configs_->activation_precision ==
      RuntimeConfigs::ActivationPrecision::kFP16) {
    flags |= XNN_FLAG_FORCE_FP16_INFERENCE;
  }
  if (runtime_configs_->xnn_profile) {
    flags |= XNN_FLAG_BASIC_PROFILING;

    if (!runtime_configs_->xnn_profile_csv.empty()) {
      MP_RETURN_IF_ERROR(mediapipe::file::SetContents(
          runtime_configs_->xnn_profile_csv, "node_id; time(us); op_name\n"));
    }
  }
  pthreadpool_t threadpool =
      pthreadpool_create(runtime_configs_->xnn_num_threads);
  threadpool_ = XnnThreadpoolPtr{threadpool, pthreadpool_destroy};

  RET_CHECK_EQ(
      xnn_status_success,
      xnn_create_runtime_v3(owned_subgraph_.get(),
                            runtime_configs_->weights_cache
                                ? runtime_configs_->weights_cache->Get()
                                : nullptr,
                            threadpool, flags, &runtime_ptr));
  RET_CHECK_NE(runtime_ptr, nullptr);
  runtime_ = XnnRuntimePtr{runtime_ptr, xnn_delete_runtime};

  return absl::OkStatus();
}

absl::Status XnnGraph::SetupRuntime() {
  {
    VLOG(3) << "input size " << input_tensors_.size();
    VLOG(3) << "output size " << output_tensors_.size();
    externals_.clear();
    // Init external
    for (const auto& input : input_tensors_) {
      VLOG(3) << "input id " << input->tensor_id(owned_subgraph_.get());
      externals_.push_back(xnn_external_value{
          input->tensor_id(owned_subgraph_.get()), input->Data()});
    }
    for (const auto& output : output_tensors_) {
      VLOG(3) << "output id " << output->tensor_id(owned_subgraph_.get());
      externals_.push_back(xnn_external_value{
          output->tensor_id(owned_subgraph_.get()), output->Data()});
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
      MP_RETURN_IF_ERROR(
          AppendStringToFile(runtime_configs_->xnn_profile_csv, ss.str()));
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Clamp(
    std::shared_ptr<Tensor> input, ClampParams params) {
  MP_ASSIGN_OR_RETURN(auto output,
                      IntermediateTensor(input->dims, "clamp_output"));

  build_steps_.push_back(
      [input, output, params](xnn_subgraph_t subgraph) -> absl::Status {
        RET_CHECK_EQ(xnn_status_success,
                     xnn_define_clamp(subgraph, params.out_min, params.out_max,
                                      input->tensor_id(subgraph),
                                      output->tensor_id(subgraph),
                                      /*flags=*/0));
        return absl::OkStatus();
      });

  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Gelu(
    std::shared_ptr<Tensor> input) {
  // x^2
  MP_ASSIGN_OR_RETURN(auto sqr_out, Square(input));

  // 0.044715 * x^2
  MP_ASSIGN_OR_RETURN(auto sqr_4471, ElementMul(sqr_out, 0.044715));

  // 1 + 0.044715 * x^2
  MP_ASSIGN_OR_RETURN(auto sqr_4471_1, ElementAdd(sqr_4471, 1.0f));

  // x + 0.044715 * x^3
  MP_ASSIGN_OR_RETURN(auto x_cube_4471, ElementMul(sqr_4471_1, input));

  constexpr float sqrt_2_over_pi = 0.7978845608;
  MP_ASSIGN_OR_RETURN(auto sqrt_2_over_pi_x_cube_4471,
                      ElementMul(x_cube_4471, sqrt_2_over_pi));

  // tanh(x + 0.044715 * x^3)
  MP_ASSIGN_OR_RETURN(auto tanh_x_cube_4471, Tanh(sqrt_2_over_pi_x_cube_4471));

  // 1 + tanh(x + 0.044715 * x^3)
  MP_ASSIGN_OR_RETURN(auto tanh_x_cube_4471_1,
                      ElementAdd(tanh_x_cube_4471, 1.0f));

  // 0.5 * (1 + [tanh(x + 0.044715 * x^3)])
  MP_ASSIGN_OR_RETURN(auto cdf, ElementMul(tanh_x_cube_4471_1, 0.5));

  return ElementMul(input, cdf);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Sigmoid(
    std::shared_ptr<Tensor> input) {
  MP_ASSIGN_OR_RETURN(auto output,
                      IntermediateTensor(input->dims, "sigmoid_output"));
  build_steps_.push_back(
      [output, input](xnn_subgraph_t subgraph) -> absl::Status {
        RET_CHECK_EQ(xnn_status_success,
                     xnn_define_sigmoid(subgraph, input->tensor_id(subgraph),
                                        output->tensor_id(subgraph),
                                        /*flags=*/0));
        return absl::Status();
      });
  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Silu(
    std::shared_ptr<Tensor> input) {
  MP_ASSIGN_OR_RETURN(auto sigmoid_out, Sigmoid(input));
  return ElementMul(input, sigmoid_out);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Relu(
    std::shared_ptr<Tensor> input) {
  return Clamp(input, {.out_min = 0});
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Relu1p5(
    std::shared_ptr<Tensor> input) {
  MP_ASSIGN_OR_RETURN(auto relu_output, Relu(input));
  MP_ASSIGN_OR_RETURN(auto sqrt_output, SquareRoot(relu_output));
  return ElementMul(relu_output, sqrt_output);
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Abs(
    std::shared_ptr<Tensor> input) {
  MP_ASSIGN_OR_RETURN(auto output,
                      IntermediateTensor(input->dims, "abs_output"));

  build_steps_.push_back(
      [input, output](xnn_subgraph_t subgraph) -> absl::Status {
        RET_CHECK_EQ(xnn_status_success,
                     xnn_define_abs(subgraph, input->tensor_id(subgraph),
                                    output->tensor_id(subgraph),
                                    /*flags=*/0));
        return absl::OkStatus();
      });
  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::Log(
    std::shared_ptr<Tensor> input) {
  MP_ASSIGN_OR_RETURN(auto output,
                      IntermediateTensor(input->dims, "log_output"));

  build_steps_.push_back(
      [input, output](xnn_subgraph_t subgraph) -> absl::Status {
        RET_CHECK_EQ(xnn_status_success,
                     xnn_define_log(subgraph, input->tensor_id(subgraph),
                                    output->tensor_id(subgraph),
                                    /*flags=*/0));
        return absl::OkStatus();
      });

  return output;
}

absl::StatusOr<std::shared_ptr<Tensor>> XnnGraphBuilder::CopySign(
    std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs) {
  MP_ASSIGN_OR_RETURN(auto output,
                      IntermediateTensor(OutDimsForElementwiseOp(*lhs, *rhs),
                                         "copysign_output"));

  build_steps_.push_back(
      [lhs, rhs, output](xnn_subgraph_t subgraph) -> absl::Status {
        RET_CHECK_EQ(xnn_status_success,
                     xnn_define_copysign(subgraph, lhs->tensor_id(subgraph),
                                         rhs->tensor_id(subgraph),
                                         output->tensor_id(subgraph),
                                         /*flags=*/0));
        return absl::OkStatus();
      });

  return output;
}

}  // namespace xnn_utils
}  // namespace mediapipe::tasks::genai

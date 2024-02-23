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

#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/discrete_weight_accessor.h"

#include <sys/stat.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/genai/inference/utils/xnn_utils/xnn_tensor.h"
#include "xnnpack.h"  // from @XNNPACK

namespace mediapipe::tasks::genai::xnn_utils {
namespace {

absl::StatusOr<int64_t> GetSize(absl::string_view name) {
  std::string name_str(name);
  struct stat stat_buf;
  RET_CHECK_EQ(stat(name_str.c_str(), &stat_buf), 0);
  return stat_buf.st_size;
}

}  // namespace

absl::StatusOr<std::shared_ptr<Tensor>>
DiscreteWeightWeightAccessor::LoadWeight(absl::string_view filename_prefix,
                                         Tensor::DimsType dims,
                                         size_t dim_scale_if_any) const {
  RET_CHECK(!filename_prefix.empty());
  RET_CHECK(!dims.empty());

  if (!mediapipe::file::IsDirectory(cache_path_).ok()) {
    MP_RETURN_IF_ERROR(mediapipe::file::RecursivelyCreateDir(cache_path_));
  }

  std::string abs_filename_prefix =
      mediapipe::file::JoinPath(weight_path_, filename_prefix);
  if (mediapipe::file::Basename(filename_prefix) == filename_prefix) {
    // In this case, the given `filename_prefix` is not from cache.
    filename_prefix = abs_filename_prefix;
  }

  if (auto s = mediapipe::file::Exists(filename_prefix); !s.ok()) {
    ABSL_DLOG(WARNING) << filename_prefix << ": " << s;
    return nullptr;
  }

  const size_t expect_num_elements = std::accumulate(
      std::begin(dims), std::end(dims), size_t(1), std::multiplies<size_t>());

  MP_ASSIGN_OR_RETURN(const int64_t file_size, GetSize(filename_prefix));
  if (file_size == expect_num_elements * sizeof(float)) {
    std::shared_ptr<Tensor> result =
        std::make_shared<Tensor>(std::move(dims), xnn_datatype_fp32);
    MP_RETURN_IF_ERROR(result->LoadFromFile(filename_prefix));
    return result;
  }
  ABSL_DLOG(INFO) << "file_size=" << file_size
                  << " expect_num_elements=" << expect_num_elements << " dims=["
                  << dims << "] file=" << filename_prefix;

  MP_RETURN_IF_ERROR(mediapipe::file::Exists(
      absl::StrCat(filename_prefix, kQuantizedScaleSuffix)));

  std::shared_ptr<Tensor> result;
  if (file_size == expect_num_elements) {
    result =
        std::make_shared<QCTensor>(dims, dim_scale_if_any, xnn_datatype_qcint8);
  } else {
    RET_CHECK_EQ(file_size, expect_num_elements / 2) << filename_prefix;
    result =
        std::make_shared<QCTensor>(dims, dim_scale_if_any, xnn_datatype_qcint4);
  }

  MP_RETURN_IF_ERROR(result->LoadFromFile(filename_prefix));

  return result;
}

absl::StatusOr<std::shared_ptr<Tensor>>
DiscreteWeightWeightAccessor::LoadTransposedWeight(
    absl::string_view filename_prefix, Tensor::DimsType original_dims,
    size_t dim_scale_if_any) const {
  RET_CHECK(!cache_path_.empty());
  auto cache_full_prefix =
      mediapipe::file::JoinPath(cache_path_, filename_prefix);
  Tensor::DimsType cache_dim{original_dims.rbegin(), original_dims.rend()};
  std::shared_ptr<Tensor> r;
  MP_ASSIGN_OR_RETURN(r, LoadWeight(cache_full_prefix, std::move(cache_dim),
                                    /*dim_scale_if_any=*/1 - dim_scale_if_any));
  if (r) {
    return r;
  }

  MP_ASSIGN_OR_RETURN(r, LoadWeight(filename_prefix, std::move(original_dims),
                                    /*dim_scale_if_any=*/dim_scale_if_any));
  if (r) {
    r = r->Transpose();
    if (auto s = r->DumpToFile(cache_full_prefix); !s.ok()) {
      return s;
    } else {
      MP_RETURN_IF_ERROR(r->LoadFromFile(cache_full_prefix));
    }
  } else {
    VLOG(2) << "Could not load " << filename_prefix;
  }
  return r;
}

}  // namespace mediapipe::tasks::genai::xnn_utils

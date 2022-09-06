/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MEDIAPIPE_TASKS_CC_CORE_TASK_API_FACTORY_H_
#define MEDIAPIPE_TASKS_CC_CORE_TASK_API_FACTORY_H_

#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/core/base_task_api.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "tensorflow/lite/core/api/op_resolver.h"

namespace mediapipe {
namespace tasks {
namespace core {

template <typename T>
using EnableIfBaseTaskApiSubclass =
    typename std::enable_if<std::is_base_of<BaseTaskApi, T>::value>::type*;

// Template creator for all subclasses of BaseTaskApi.
class TaskApiFactory {
 public:
  TaskApiFactory() = delete;

  template <typename T, typename Options,
            EnableIfBaseTaskApiSubclass<T> = nullptr>
  static absl::StatusOr<std::unique_ptr<T>> Create(
      CalculatorGraphConfig graph_config,
      std::unique_ptr<tflite::OpResolver> resolver,
      PacketsCallback packets_callback = nullptr) {
    bool found_task_subgraph = false;
    for (const auto& node : graph_config.node()) {
      if (node.calculator() == "FlowLimiterCalculator") {
        continue;
      }
      if (found_task_subgraph) {
        return CreateStatusWithPayload(
            absl::StatusCode::kInvalidArgument,
            "Task graph config should only contain one task subgraph node.",
            MediaPipeTasksStatus::kInvalidTaskGraphConfigError);
      } else {
        if (!node.options().HasExtension(Options::ext)) {
          return CreateStatusWithPayload(
              absl::StatusCode::kInvalidArgument,
              absl::StrCat(node.calculator(),
                           " is missing the required task options field."),
              MediaPipeTasksStatus::kInvalidTaskGraphConfigError);
        }
        found_task_subgraph = true;
      }
    }
    ASSIGN_OR_RETURN(
        auto runner,
        core::TaskRunner::Create(std::move(graph_config), std::move(resolver),
                                 std::move(packets_callback)));
    return std::make_unique<T>(std::move(runner));
  }
};

}  // namespace core
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_CORE_TASK_API_FACTORY_H_

// Copyright 2019 The MediaPipe Authors.
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

#include "mediapipe/framework/tool/executor_util.h"

#include <string>

#include "mediapipe/framework/mediapipe_options.pb.h"
#include "mediapipe/framework/thread_pool_executor.pb.h"

namespace mediapipe {
namespace tool {

void EnsureMinimumDefaultExecutorStackSize(const int32_t min_stack_size,
                                           CalculatorGraphConfig* config) {
  mediapipe::ExecutorConfig* default_executor_config = nullptr;
  for (mediapipe::ExecutorConfig& executor_config :
       *config->mutable_executor()) {
    if (executor_config.name().empty()) {
      default_executor_config = &executor_config;
      break;
    }
  }
  if (!default_executor_config) {
    default_executor_config = config->add_executor();
    if (config->num_threads()) {
      default_executor_config->mutable_options()
          ->MutableExtension(mediapipe::ThreadPoolExecutorOptions::ext)
          ->set_num_threads(config->num_threads());
      config->clear_num_threads();
    }
  }
  if (default_executor_config->type().empty() ||
      default_executor_config->type() == "ThreadPoolExecutor") {
    mediapipe::ThreadPoolExecutorOptions* extension =
        default_executor_config->mutable_options()->MutableExtension(
            mediapipe::ThreadPoolExecutorOptions::ext);
    if (extension->stack_size() < min_stack_size) {
      extension->set_stack_size(min_stack_size);
    }
  }
}

}  // namespace tool
}  // namespace mediapipe

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

#ifndef MEDIAPIPE_GPU_GPU_TEST_BASE_H_
#define MEDIAPIPE_GPU_GPU_TEST_BASE_H_

#include <functional>
#include <memory>

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"

namespace mediapipe {

class GpuTestEnvironment {
 protected:
  GpuTestEnvironment() { helper_.InitializeForTest(gpu_resources_.get()); }

  void RunInGlContext(std::function<void(void)> gl_func) {
    helper_.RunInGlContext(std::move(gl_func));
  }

  GpuSharedData gpu_shared_;
  std::shared_ptr<GpuResources> gpu_resources_ = gpu_shared_.gpu_resources;
  GlCalculatorHelper helper_;
};

class GpuTestBase : public testing::Test, public GpuTestEnvironment {};

template <typename T>
class GpuTestWithParamBase : public testing::TestWithParam<T>,
                             public GpuTestEnvironment {};

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_GPU_TEST_BASE_H_

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

// Declares GpuSharedData, a private object that is used to store
// platform-specific resources shared by GPU calculators across a graph.

// Consider this file an implementation detail. None of this is part of the
// public API.

#ifndef MEDIAPIPE_GPU_GPU_SHARED_DATA_INTERNAL_H_
#define MEDIAPIPE_GPU_GPU_SHARED_DATA_INTERNAL_H_

#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_node.h"
#include "mediapipe/framework/executor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gl_context.h"
#include "mediapipe/gpu/gpu_buffer_multi_pool.h"

#ifdef __APPLE__
#ifdef __OBJC__
@class MPPGraphGPUData;
#else
struct MPPGraphGPUData;
#endif  // __OBJC__
#endif  // defined(__APPLE__)

namespace mediapipe {

// TODO: rename to GpuService or GpuManager or something.
class GpuResources {
 public:
  using StatusOrGpuResources =
      ::mediapipe::StatusOr<std::shared_ptr<GpuResources>>;

  static StatusOrGpuResources Create();
  static StatusOrGpuResources Create(PlatformGlContext external_context);

  // The destructor must be defined in the implementation file so that on iOS
  // the correct ARC release calls are generated.
  ~GpuResources();

  explicit GpuResources(PlatformGlContext external_context);

  // Shared GL context for calculators.
  // TODO: require passing a context or node identifier.
  const std::shared_ptr<GlContext>& gl_context() {
    return gl_context(nullptr);
  };

  const std::shared_ptr<GlContext>& gl_context(CalculatorContext* cc);

  // Shared buffer pool.
  GpuBufferMultiPool& gpu_buffer_pool() { return gpu_buffer_pool_; }

#ifdef __APPLE__
  MPPGraphGPUData* ios_gpu_data();
#endif  // defined(__APPLE__)

  void PrepareGpuNode(CalculatorNode* node);

  // If the node requires custom GPU executors in the current configuration,
  // returns the executor's names and the executors themselves.
  const std::map<std::string, std::shared_ptr<Executor>>& GetGpuExecutors() {
    return named_executors_;
  }

 private:
  GpuResources() = delete;
  explicit GpuResources(std::shared_ptr<GlContext> gl_context);

  const std::shared_ptr<GlContext>& gl_context(const std::string& key);
  const std::string& ContextKey(const std::string& canonical_node_name);

  std::map<std::string, std::string> node_key_;
  std::map<std::string, std::shared_ptr<GlContext>> gl_key_context_;

  // The pool must be destructed before the gl_context, but after the
  // ios_gpu_data, so the declaration order is important.
  GpuBufferMultiPool gpu_buffer_pool_;

#ifdef __APPLE__
  // Note that this is an Objective-C object.
  MPPGraphGPUData* ios_gpu_data_;
#endif  // defined(__APPLE__)

  std::map<std::string, std::shared_ptr<Executor>> named_executors_;
};

// Legacy struct to keep existing client code happy.
// TODO: eliminate!
struct GpuSharedData {
  GpuSharedData();

  explicit GpuSharedData(PlatformGlContext external_context)
      : GpuSharedData(CreateGpuResourcesOrDie(external_context)) {}

  explicit GpuSharedData(std::shared_ptr<GpuResources> gpu_resources)
      : gpu_resources(gpu_resources),
        gl_context(gpu_resources->gl_context()),
        gpu_buffer_pool(gpu_resources->gpu_buffer_pool()) {}

  std::shared_ptr<GpuResources> gpu_resources;

  std::shared_ptr<GlContext> gl_context;

  GpuBufferMultiPool& gpu_buffer_pool;

 private:
  static std::shared_ptr<GpuResources> CreateGpuResourcesOrDie(
      PlatformGlContext external_context) {
    auto status_or_resources = GpuResources::Create(external_context);
    MEDIAPIPE_CHECK_OK(status_or_resources.status())
        << ": could not create GpuResources";
    return std::move(status_or_resources).ValueOrDie();
  }
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_GPU_SHARED_DATA_INTERNAL_H_

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

#include "mediapipe/gpu/gpu_shared_data_internal.h"

#include "mediapipe/framework/deps/no_destructor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/gpu/gl_context.h"
#include "mediapipe/gpu/gl_context_options.pb.h"
#include "mediapipe/gpu/graph_support.h"

#if __APPLE__
#import "mediapipe/gpu/MPPGraphGPUData.h"
#endif  // __APPLE__

namespace mediapipe {

#if __APPLE__
static constexpr bool kGlContextUseDedicatedThread = false;
#elif defined(__EMSCRIPTEN__)
// Since we're forcing single-threaded execution, we just run everything
// in-place.
static constexpr bool kGlContextUseDedicatedThread = false;
#else
// TODO: in theory this is only needed on Android. In practice,
// when using SwiftShader on Linux, we get memory leaks if we attempt to get
// the current GL context on a random thread. For now let's keep the
// single-thread approach on Linux as a workaround.
static constexpr bool kGlContextUseDedicatedThread = true;
#endif  // __APPLE__

// If true, use a single GL context shared by all calculators.
// If false, create a separate context per calculator.
// Context-per-calculator (i.e. setting this to false) is not fully supported,
// and it is only known to work on iOS.
static constexpr bool kGlCalculatorShareContext = true;

// Allow a GlContext to be used as an Executor. This makes it possible to run
// GPU-based calculators directly on the GlContext thread, avoiding two thread
// switches.
class GlContextExecutor : public Executor {
 public:
  explicit GlContextExecutor(GlContext* gl_context) : gl_context_(gl_context) {}
  ~GlContextExecutor() override {}
  void Schedule(std::function<void()> task) override {
    gl_context_->RunWithoutWaiting(std::move(task));
  }

 private:
  GlContext* const gl_context_;
};

static const std::string& SharedContextKey() {
  static const mediapipe::NoDestructor<std::string> kSharedContextKey("");
  return *kSharedContextKey;
}

GpuResources::StatusOrGpuResources GpuResources::Create() {
  return Create(kPlatformGlContextNone);
}

GpuResources::StatusOrGpuResources GpuResources::Create(
    PlatformGlContext external_context) {
  ASSIGN_OR_RETURN(
      std::shared_ptr<GlContext> context,
      GlContext::Create(external_context, kGlContextUseDedicatedThread));
  std::shared_ptr<GpuResources> gpu_resources(
      new GpuResources(std::move(context)));
  return gpu_resources;
}

GpuResources::GpuResources(std::shared_ptr<GlContext> gl_context) {
  gl_key_context_[SharedContextKey()] = gl_context;
  named_executors_[kGpuExecutorName] =
      std::make_shared<GlContextExecutor>(gl_context.get());
#if __APPLE__
  gpu_buffer_pool().RegisterTextureCache(gl_context->cv_texture_cache());
  ios_gpu_data_ = [[MPPGraphGPUData alloc] initWithContext:gl_context.get()
                                                 multiPool:&gpu_buffer_pool_];
#endif  // __APPLE__
}

GpuResources::~GpuResources() {
#if __APPLE__
  // Note: on Apple platforms, this object contains Objective-C objects. The
  // destructor will release them, but ARC must be on.
#if !__has_feature(objc_arc)
#error This file must be built with ARC.
#endif
  for (auto& kv : gl_key_context_) {
    gpu_buffer_pool().UnregisterTextureCache(kv.second->cv_texture_cache());
  }
#endif
}

absl::Status GpuResources::PrepareGpuNode(CalculatorNode* node) {
  CHECK(ContainsKey(node->Contract().ServiceRequests(), kGpuService.key));
  std::string node_id = node->GetCalculatorState().NodeName();
  std::string node_type = node->GetCalculatorState().CalculatorType();
  std::string context_key;

#ifndef __EMSCRIPTEN__
  // TODO Allow calculators to request a separate context.
  // For now, allow a few calculators to run in their own context.
  bool gets_own_context = (node_type == "ImageFrameToGpuBufferCalculator") ||
                          (node_type == "GpuBufferToImageFrameCalculator") ||
                          (node_type == "GlSurfaceSinkCalculator");

  const auto& options =
      node->GetCalculatorState().Options<mediapipe::GlContextOptions>();
  if (options.has_gl_context_name() && !options.gl_context_name().empty()) {
    context_key = absl::StrCat("user:", options.gl_context_name());
  } else if (gets_own_context) {
    context_key = absl::StrCat("auto:", node_type);
  } else if (kGlCalculatorShareContext) {
    context_key = SharedContextKey();
  } else {
    context_key = absl::StrCat("auto:", node_id);
  }
#else
  // On Emscripten we currently do not support multiple contexts.
  context_key = SharedContextKey();
#endif  // !__EMSCRIPTEN__
  node_key_[node_id] = context_key;

  ASSIGN_OR_RETURN(std::shared_ptr<GlContext> context,
                   GetOrCreateGlContext(context_key));

  if (kGlContextUseDedicatedThread) {
    std::string executor_name =
        absl::StrCat(kGpuExecutorName, "_", context_key);
    node->SetExecutor(executor_name);
    if (!ContainsKey(named_executors_, executor_name)) {
      named_executors_.emplace(
          executor_name, std::make_shared<GlContextExecutor>(context.get()));
    }
  }
  context->SetProfilingContext(
      node->GetCalculatorState().GetSharedProfilingContext());

  return OkStatus();
}

// TODO: expose and use an actual ID instead of using the
// canonicalized name.
const std::shared_ptr<GlContext>& GpuResources::gl_context(
    CalculatorContext* cc) {
  if (cc) {
    auto it = gl_key_context_.find(node_key_[cc->NodeName()]);
    if (it != gl_key_context_.end()) {
      return it->second;
    }
  }

  return gl_key_context_[SharedContextKey()];
}

GlContext::StatusOrGlContext GpuResources::GetOrCreateGlContext(
    const std::string& key) {
  auto it = gl_key_context_.find(key);
  if (it == gl_key_context_.end()) {
    ASSIGN_OR_RETURN(std::shared_ptr<GlContext> new_context,
                     GlContext::Create(*gl_key_context_[SharedContextKey()],
                                       kGlContextUseDedicatedThread));
    it = gl_key_context_.emplace(key, new_context).first;
#if __APPLE__
    gpu_buffer_pool_.RegisterTextureCache(it->second->cv_texture_cache());
#endif
  }
  return it->second;
}

GpuSharedData::GpuSharedData() : GpuSharedData(kPlatformGlContextNone) {}

#if __APPLE__
MPPGraphGPUData* GpuResources::ios_gpu_data() { return ios_gpu_data_; }
#endif  // __APPLE__

}  // namespace mediapipe

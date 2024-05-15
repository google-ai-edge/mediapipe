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

#include <memory>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/deps/no_destructor.h"
#include "mediapipe/gpu/gl_context.h"
#include "mediapipe/gpu/gl_context_options.pb.h"
#include "mediapipe/gpu/graph_support.h"
#include "mediapipe/gpu/multi_pool.h"

#if __APPLE__
#include "mediapipe/gpu/metal_shared_resources.h"
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
  ~GlContextExecutor() override = default;
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
    PlatformGlContext external_context,
    const MultiPoolOptions* gpu_buffer_pool_options) {
  MP_ASSIGN_OR_RETURN(
      std::shared_ptr<GlContext> context,
      GlContext::Create(external_context, kGlContextUseDedicatedThread));
  std::shared_ptr<GpuResources> gpu_resources(
      new GpuResources(std::move(context), gpu_buffer_pool_options));
  return gpu_resources;
}

GpuResources::GpuResources(std::shared_ptr<GlContext> gl_context,
                           const MultiPoolOptions* gpu_buffer_pool_options)
    : gl_key_context_(new GlContextMapType(),
                      [](auto* map) {
                        // This flushes all pending jobs in all GL contexts,
                        // ensuring that all GL contexts not referenced
                        // elsewhere are destroyed as part of this destructor.
                        // Failure to do this may cause GL threads to outlast
                        // this destructor and execute jobs after the
                        // GpuResources object is destroyed.
                        for (auto& [key, context] : *map) {
                          const auto status = std::move(context)->Run(
                              []() { return absl::OkStatus(); });
                          ABSL_LOG_IF(ERROR, !status.ok())
                              << "Failed to flush GlContext jobs: " << status;
                        }
                        delete map;
                      }),
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
      texture_caches_(std::make_shared<CvTextureCacheManager>()),
      gpu_buffer_pool_(
          [tc = texture_caches_](const internal::GpuBufferSpec& spec,
                                 const MultiPoolOptions& options) {
            return CvPixelBufferPoolWrapper::Create(spec, options, tc.get());
          },
          gpu_buffer_pool_options ? *gpu_buffer_pool_options
                                  : kDefaultMultiPoolOptions)
#else   // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
          gpu_buffer_pool_(gpu_buffer_pool_options ? *gpu_buffer_pool_options
                                            : kDefaultMultiPoolOptions)
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
{
  gl_key_context_->insert({SharedContextKey(), gl_context});
  named_executors_[kGpuExecutorName] =
      std::make_shared<GlContextExecutor>(gl_context.get());
#if __APPLE__
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  texture_caches_->RegisterTextureCache(gl_context->cv_texture_cache());
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  metal_shared_ = std::make_unique<MetalSharedResources>();
#endif  // __APPLE__
}

GpuResources::~GpuResources() {
  // This flushes all pending jobs in all GL contexts,
  // ensuring that all existing jobs, which may refer GpuResource and kept their
  // gpu resources (e.g. GpuResources::gpu_buffer_pool_) through a raw pointer,
  // have finished before kept gpu resources get deleted.
  for (auto& [key, context] : *gl_key_context_) {
    const auto status = context->Run([]() { return absl::OkStatus(); });
    ABSL_LOG_IF(ERROR, !status.ok())
        << "Failed to flush GlContext jobs: " << status;
  }
#if __APPLE__
  // Note: on Apple platforms, this object contains Objective-C objects.
  // The destructor will release them, but ARC must be on.
#if !__has_feature(objc_arc)
#error This file must be built with ARC.
#endif
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  for (auto& kv : *gl_key_context_) {
    texture_caches_->UnregisterTextureCache(kv.second->cv_texture_cache());
  }
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
#endif  // __APPLE__
}

ABSL_CONST_INIT extern const GraphService<GpuResources> kGpuService;

absl::Status GpuResources::PrepareGpuNode(CalculatorNode* node) {
  ABSL_CHECK(node->Contract().ServiceRequests().contains(kGpuService.key));
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

  MP_ASSIGN_OR_RETURN(std::shared_ptr<GlContext> context,
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
    auto it = gl_key_context_->find(node_key_[cc->NodeName()]);
    if (it != gl_key_context_->end()) {
      return it->second;
    }
  }

  return gl_key_context_->at(SharedContextKey());
}

GlContext::StatusOrGlContext GpuResources::GetOrCreateGlContext(
    const std::string& key) {
  auto it = gl_key_context_->find(key);
  if (it == gl_key_context_->end()) {
    MP_ASSIGN_OR_RETURN(
        std::shared_ptr<GlContext> new_context,
        GlContext::Create(*gl_key_context_->at(SharedContextKey()),
                          kGlContextUseDedicatedThread));
    it = gl_key_context_->emplace(key, new_context).first;
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
    texture_caches_->RegisterTextureCache(it->second->cv_texture_cache());
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  }
  return it->second;
}

GpuSharedData::GpuSharedData() : GpuSharedData(kPlatformGlContextNone) {}

#if !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
static std::shared_ptr<GlTextureBuffer> GetGlTextureBufferFromPool(
    int width, int height, GpuBufferFormat format) {
  std::shared_ptr<GlTextureBuffer> texture_buffer;
  const auto cc = LegacyCalculatorSupport::Scoped<CalculatorContext>::current();

  if (cc && cc->Service(kGpuService).IsAvailable()) {
    GpuBufferMultiPool* pool =
        &cc->Service(kGpuService).GetObject().gpu_buffer_pool();
    // Note that the "gpu_buffer_pool" serves GlTextureBuffers on non-Apple
    // platforms. TODO: refactor into storage pools.
    auto texture_buffer_from_pool = pool->GetBuffer(width, height, format);
    ABSL_CHECK_OK(texture_buffer_from_pool);
    texture_buffer =
        texture_buffer_from_pool->internal_storage<GlTextureBuffer>();
  } else {
    texture_buffer = GlTextureBuffer::Create(width, height, format);
  }
  return texture_buffer;
}

static auto kGlTextureBufferPoolRegistration = [] {
  // Ensure that the GlTextureBuffer's own factory is already registered, so we
  // can override it.
  GlTextureBuffer::RegisterOnce();
  return internal::GpuBufferStorageRegistry::Get()
      .RegisterFactory<GlTextureBuffer>(GetGlTextureBufferFromPool);
}();
#endif  // !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

}  // namespace mediapipe

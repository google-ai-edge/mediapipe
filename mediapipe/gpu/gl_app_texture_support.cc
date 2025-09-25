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

#include "mediapipe/gpu/gl_app_texture_support.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/executor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gl_context.h"
#include "mediapipe/gpu/gl_texture_buffer.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include "mediapipe/gpu/multi_pool.h"

namespace mediapipe {

absl::Status SetExternalGlContextForGraph(CalculatorGraph* graph,
                                          PlatformGlContext external_context) {
  MP_ASSIGN_OR_RETURN(auto gpu_resources,
                      GpuResources::Create(external_context));
  return graph->SetGpuResources(std::move(gpu_resources));
}

absl::StatusOr<std::shared_ptr<GpuResources>> CreateGpuResources(
    PlatformGlContext external_context /*=nullptr*/,
    const MultiPoolOptions* gpu_buffer_pool_options /*=nullptr*/) {
  return GpuResources::Create(external_context, gpu_buffer_pool_options);
}

absl::StatusOr<std::shared_ptr<Executor>> GetDefaultGpuExecutor(
    const GpuResources& gpu_resources) {
  return gpu_resources.GetDefaultGpuExecutor();
}

absl::StatusOr<GpuBuffer> WrapExternalGlTexture(
    const GpuResources& gpu_resources, GLenum target, GLuint name, int width,
    int height, GpuBufferFormat format,
    GlTextureBuffer::DeletionCallback release_callback,
    WrapExternalGlTextureSyncMode sync_mode) {
  auto& gl_context = gpu_resources.gl_context();
  auto buffer = GlTextureBuffer::Wrap(target, name, width, height, format,
                                      gl_context, release_callback);

  if (sync_mode == WrapExternalGlTextureSyncMode::kNoSync) {
    return GpuBuffer(std::move(buffer));
  }

  auto sync = GlContext::CreateSyncTokenForCurrentExternalContext(gl_context);
  if (sync) {
    buffer->Updated(sync);
  } else if (sync_mode == WrapExternalGlTextureSyncMode::kSync) {
    return absl::InternalError("Failed to create a sync.");
  }
  return GpuBuffer(std::move(buffer));
}

absl::StatusOr<Packet> WrapExternalGlTextureForGraph(
    const CalculatorGraph& graph, GLenum target, GLuint name, int width,
    int height, GpuBufferFormat format,
    GlTextureBuffer::DeletionCallback release_callback, bool skip_input_sync) {
  const std::shared_ptr<GpuResources> gpu_resources = graph.GetGpuResources();
  RET_CHECK(gpu_resources)
      << "Cannot wrap external GlTexture for the the graph which is not "
         "configured with GpuResources.";
  MP_ASSIGN_OR_RETURN(
      GpuBuffer gpu_buffer,
      WrapExternalGlTexture(
          *gpu_resources, target, name, width, height, format,
          std::move(release_callback),
          skip_input_sync ? WrapExternalGlTextureSyncMode::kNoSync
                          : WrapExternalGlTextureSyncMode::kMaybeSyncOrFinish));
  return MakePacket<GpuBuffer>(std::move(gpu_buffer));
}

}  // namespace mediapipe

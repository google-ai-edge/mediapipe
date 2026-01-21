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

#ifndef MEDIAPIPE_GPU_GL_APP_TEXTURE_SUPPORT_H_
#define MEDIAPIPE_GPU_GL_APP_TEXTURE_SUPPORT_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/executor.h"
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gl_context.h"
#include "mediapipe/gpu/gl_texture_buffer.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/gpu/multi_pool.h"

namespace mediapipe {

class GpuResources;

// Sets an OpenGL context that will share resources with the graph.
// This is necessary in order to send GL textures into the graph, or receive
// them from the graph.
// Call this before starting the graph.
//
// Usage example:
//   (assuming the desired GL context is current on this thread)
//   MP_RETURN_IF_ERROR(mediapipe::SetExternalGlContextForGraph(
//       &graph, mediapipe::GlContext::GetCurrentNativeContext()));
absl::Status SetExternalGlContextForGraph(CalculatorGraph* graph,
                                          PlatformGlContext external_context);

// Creates GPU resources for a graph using a platform external context. If
// external_context is not kPlatformGlContextNone, then all shareable
// data in the context is shared. gpu_buffer_pool_options is an optional
// parameter to specify the options for pooling GpuBuffer objects.
// Call this after initializing a graph and before start running it.
absl::StatusOr<std::shared_ptr<GpuResources>> CreateGpuResources(
    PlatformGlContext external_context = kPlatformGlContextNone,
    const MultiPoolOptions* gpu_buffer_pool_options = nullptr);

// Gets the default GPU executor that will be used by calculators requested
// GpuService (a.k.a. GpuResources).
//
// Might be useful if you want to force all calculators to execute on default
// GPU executor by setting it on CalculatorGraph::SetExecutor("", ...);
//
// Note: alternatively, you can also initialize your own GL context and use
// ApplicationThreadExecutor on the calculator graph if executing graph on a
// calling thread is appropriate.
absl::StatusOr<std::shared_ptr<Executor>> GetDefaultGpuExecutor(
    const GpuResources& gpu_resources);

// Wraps an external OpenGL texture into a GpuBuffer packet that can be sent
// into a MediaPipe graph.
//
// `release_callback` is a callback that will be called when MediaPipe is done
// with the texture. It is passed a GlSyncToken that should be waited upon to
// ensure the GPU processing using the texture is done. In other words, the
// callback is used to signal that the CPU is done with the texture, and the
// token is used to ensure the GPU is also done. These two phases are kept
// separate to avoid unnecessary CPU/GPU synchronization.
//
// If the application uses other mechanisms to ensure processing is complete
// (e.g. WaitUntilIdle and glFinish), then it can pass nullptr.
//
// `skip_input_sync` should normally be set to false. You can set it to true if
// the texture's contents are guaranteed to be already visible to any context
// (e.g. if you have called glFinish).
//
// Usage example:
//   MP_ASSIGN_OR_RETURN(
//       mediapipe::Packet packet,
//       mediapipe::WrapExternalGlTextureForGraph(
//           graph, GL_TEXTURE_2D, tex_id, tex_width, tex_height,
//           mediapipe::GpuBufferFormat::kBGRA32,
//           /*release_callback=*/nullptr));
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
ABSL_DEPRECATED("Prefer using CVPixelBufferRef on Apple platforms")
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
absl::StatusOr<Packet> WrapExternalGlTextureForGraph(
    const CalculatorGraph& graph, GLenum target, GLuint name, int width,
    int height, GpuBufferFormat format,
    GlTextureBuffer::DeletionCallback release_callback,
    bool skip_input_sync = false);

enum class WrapExternalGlTextureSyncMode {
  // External texture is already up-to-date and can be used on a shared context
  // as is (e.g. prior glFinish call) or there's just a single GL context used
  // for both external textures and MediaPipe graph.
  kNoSync,
  // MediaPipe graph has a dedicated GL context(s) and external texture must be
  // efficiently synchronized using GL sync object.
  kSync,
  // MediaPipe graph has a dedicated GL context(s) and external texture can be
  // synchronized using GL sync object or glFinish or can be skipped
  // alltogether.
  kMaybeSyncOrFinish,
};

// Wraps an external OpenGL texture into a GpuBuffer packet that can be sent
// into one or multiple MediaPipe graphs using/sharing same `GpuResources`.
//
// `release_callback` is a callback that will be called when MediaPipe is done
// with the texture. It is passed a GlSyncToken that should be waited upon to
// ensure the GPU processing using the texture is done. In other words, the
// callback is used to signal that the CPU is done with the texture, and the
// token is used to ensure the GPU is also done. These two phases are kept
// separate to avoid unnecessary CPU/GPU synchronization.
//
// Similar to `WrapExternalGlTextureForGraph` function, but allows to request a
// fine grained synchronization mode using `WrapExternalGlTextureSyncMode`.
//
// For example: requiring efficient synchronization and failing otherwise, where
// the above function can skip synchronization alltogether if it is invoked
// without external context being current on the calling thread.
//
// NOTE: returns GpuBuffer which can be wrapped into a packet as
//   `MakePacket<GpuBuffer>(std::move(gpu_buffer))`.
absl::StatusOr<GpuBuffer> WrapExternalGlTexture(
    const GpuResources& gpu_resources, GLenum target, GLuint name, int width,
    int height, GpuBufferFormat format,
    GlTextureBuffer::DeletionCallback release_callback,
    WrapExternalGlTextureSyncMode sync_mode);

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_GL_APP_TEXTURE_SUPPORT_H_

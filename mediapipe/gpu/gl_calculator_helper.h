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

#ifndef MEDIAPIPE_GPU_GL_CALCULATOR_HELPER_H_
#define MEDIAPIPE_GPU_GL_CALCULATOR_HELPER_H_

#include <memory>

#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_contract.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/packet_set.h"
#include "mediapipe/framework/packet_type.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gl_context.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/graph_support.h"
#ifdef __APPLE__
#include <CoreVideo/CoreVideo.h>

#include "mediapipe/objc/CFHolder.h"
#endif  // __APPLE__

namespace mediapipe {

class GlCalculatorHelperImpl;
class GlTexture;
class GpuResources;
struct GpuSharedData;

#ifdef __APPLE__
#if TARGET_OS_OSX
typedef CVOpenGLTextureRef CVTextureType;
#else
typedef CVOpenGLESTextureRef CVTextureType;
#endif  // TARGET_OS_OSX
#endif  // __APPLE__

// TODO: remove this and Process below, or make Process available
// on Android.
typedef std::function<void(const GlTexture& src, const GlTexture& dst)>
    RenderFunction;

// Helper class that manages OpenGL contexts and operations.
// Calculators that implement an image filter, taking one input stream of
// frames and producing one output stream of frame, should subclass
// GlSimpleCalculatorBase instead of using GlCalculatorHelper directly.
// Direct use of this class is recommended for calculators that do not fit
// that mold (e.g. calculators that combine two video streams).
class GlCalculatorHelper {
 public:
  GlCalculatorHelper();
  ~GlCalculatorHelper();

  // Call Open from the Open method of a calculator to initialize the helper.
  ::mediapipe::Status Open(CalculatorContext* cc);

  // Can be used to initialize the helper outside of a calculator. Useful for
  // testing.
  void InitializeForTest(GpuResources* gpu_resources);
  void InitializeForTest(GpuSharedData* gpu_shared);

  // This method can be called from GetContract to set up the needed GPU
  // resources.
  static ::mediapipe::Status UpdateContract(CalculatorContract* cc);

  // This method can be called from FillExpectations to set the correct types
  // for the shared GL input side packet(s).
  static ::mediapipe::Status SetupInputSidePackets(
      PacketTypeSet* input_side_packets);

  // Execute the provided function within the helper's GL context. On some
  // platforms, this may be run on a different thread; however, this method
  // will still wait for the function to finish executing before returning.
  // The status result from the function is passed on to the caller.
  ::mediapipe::Status RunInGlContext(
      std::function<::mediapipe::Status(void)> gl_func);

  // Convenience version of RunInGlContext for arguments with a void result
  // type. As with the ::mediapipe::Status version, this also waits for the
  // function to finish executing before returning.
  //
  // Implementation note: we cannot use a std::function<void(void)> argument
  // here, because that would break passing in a lambda that returns a status;
  // e.g.:
  //   RunInGlContext([]() -> ::mediapipe::Status { ... });
  //
  // The reason is that std::function<void(...)> allows the implicit conversion
  // of a callable with any result type, as long as the argument types match.
  // As a result, the above lambda would be implicitly convertible to both
  // std::function<::mediapipe::Status(void)> and std::function<void(void)>, and
  // the invocation would be ambiguous.
  //
  // Therefore, instead of using std::function<void(void)>, we use a template
  // that only accepts arguments with a void result type.
  template <typename T, typename = typename std::enable_if<std::is_void<
                            typename std::result_of<T()>::type>::value>::type>
  void RunInGlContext(T f) {
    RunInGlContext([f] {
      f();
      return ::mediapipe::OkStatus();
    }).IgnoreError();
  }

  // Use CreateSourceTexture and CreateDestinationTexture to set up textures
  // for input and output frames. They are not just a convenience; on platforms
  // where it is supported (iOS, for now) they take advantage of memory sharing
  // between the CPU and GPU, avoiding memory copies.

  // Creates a texture representing an input frame.
  GlTexture CreateSourceTexture(const GpuBuffer& pixel_buffer);
  GlTexture CreateSourceTexture(const ImageFrame& image_frame);

#ifdef __APPLE__
  // Creates a texture from a plane of a planar buffer.
  // The plane index is zero-based. The number of planes depends on the
  // internal format of the buffer.
  GlTexture CreateSourceTexture(const GpuBuffer& pixel_buffer, int plane);
#endif

  // Extracts GpuBuffer dimensions without creating a texture.
  ABSL_DEPRECATED("Use width and height methods on GpuBuffer instead")
  void GetGpuBufferDimensions(const GpuBuffer& pixel_buffer, int* width,
                              int* height);

  // Creates a texture representing an output frame.
  // TODO: This should either return errors or a status.
  GlTexture CreateDestinationTexture(
      int output_width, int output_height,
      GpuBufferFormat format = GpuBufferFormat::kBGRA32);

  // The OpenGL name of the output framebuffer.
  GLuint framebuffer() const;

  // Binds the rendering framebuffer to a destination texture.
  // TODO: do we need an unbind method too?
  void BindFramebuffer(const GlTexture& dst);

  GlContext& GetGlContext() const;

 private:
  std::unique_ptr<GlCalculatorHelperImpl> impl_;
};

// Represents an OpenGL texture.
class GlTexture {
 public:
  GlTexture() {}
  GlTexture(GLuint name, int width, int height);

  ~GlTexture() { Release(); }

  int width() const { return width_; }
  int height() const { return height_; }
  GLenum target() const { return target_; }
  GLuint name() const { return name_; }

  // Returns a buffer that can be sent to another calculator.
  // Can be used with GpuBuffer or ImageFrame.
  template <typename T>
  std::unique_ptr<T> GetFrame() const;

  // Releases texture memory
  void Release();

 private:
  friend class GlCalculatorHelperImpl;
  GlCalculatorHelperImpl* helper_impl_ = nullptr;
  GLuint name_ = 0;
  int width_ = 0;
  int height_ = 0;
  GLenum target_ = GL_TEXTURE_2D;

#ifdef MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  // For CVPixelBufferRef-based rendering
  CFHolder<CVTextureType> cv_texture_;
#else
  // Keeps track of whether this texture mapping is for read access, so that
  // we can create a consumer sync point when releasing it.
  bool for_reading_ = false;
#endif
  GpuBuffer gpu_buffer_;
  int plane_ = 0;
};

// Returns the entry with the given tag if the collection uses tags, with the
// given index otherwise. Can be used with PacketTypeSet*, PacketSet,
// OutputStreamSet, InputStreamSet, etc.
// It would be possible to have a single version of this if we could use
// non-const references. Unfortunately, they are not allowed by the style guide.
// The const-reference version cannot work with PacketTypeSet because the Set
// method is (naturally) non-const. We could add a const_cast, but I figure
// it is better to keep const-safety and accept having two versions of the
// same thing.
template <typename T>
auto TagOrIndex(const T& collection, const std::string& tag, int index)
    -> decltype(collection.Tag(tag)) {
  return collection.UsesTags() ? collection.Tag(tag) : collection.Index(index);
}

template <typename T>
auto TagOrIndex(T* collection, const std::string& tag, int index)
    -> decltype(collection->Tag(tag)) {
  return collection->UsesTags() ? collection->Tag(tag)
                                : collection->Index(index);
}

template <typename T>
bool HasTagOrIndex(const T& collection, const std::string& tag, int index) {
  return collection.UsesTags() ? collection.HasTag(tag)
                               : index < collection.NumEntries();
}

template <typename T>
bool HasTagOrIndex(T* collection, const std::string& tag, int index) {
  return collection->UsesTags() ? collection->HasTag(tag)
                                : index < collection->NumEntries();
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_GL_CALCULATOR_HELPER_H_

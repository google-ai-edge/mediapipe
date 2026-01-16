#ifndef MEDIAPIPE_GPU_GPU_BUFFER_TO_IMAGE_FRAME_CALCULATOR_H_
#define MEDIAPIPE_GPU_GPU_BUFFER_TO_IMAGE_FRAME_CALCULATOR_H_

#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/api3/one_of.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/gpu/gpu_buffer.h"

namespace mediapipe::api3 {

// Converts an input image (GpuBuffer or ImageFrame) to ImageFrame.
//
// NOTE: all GpuBufferToImageFrameCalculators use a common dedicated shared GL
// context thread by default, which is different from the main GL context thread
// used by the graph. (If MediaPipe uses multithreading and multiple OpenGL
// contexts.)
//
// IMPORTANT: graph writer must make sure input GpuBuffer backed OpenGL texture
// is not in use before the calculator starts processing and not used by any
// other code until the calculator returns:
// - pixel transfer involves attaching GpuBuffer backing texture as a logical
//   buffer to a particular bound framebuffer.
// - and if texture is already bound and enabled for texturing, this may lead
//   to a "feedback loop" and undefined results.
// See, OpenGL ES 3.0 Spec 4.4.3 "Feedback Loops between Textures and the
// Framebuffer"
//
struct GpuBufferToImageFrameNode : Node<"GpuBufferToImageFrameCalculator"> {
  template <typename S>
  struct Contract {
    Input<S, OneOf<GpuBuffer, ImageFrame>> in{""};
    Output<S, ImageFrame> out{""};
  };
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_GPU_GPU_BUFFER_TO_IMAGE_FRAME_CALCULATOR_H_

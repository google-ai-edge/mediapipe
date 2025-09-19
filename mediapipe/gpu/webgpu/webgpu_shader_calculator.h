#ifndef MEDIAPIPE_GPU_WEBGPU_WEBGPU_SHADER_CALCULATOR_H_
#define MEDIAPIPE_GPU_WEBGPU_WEBGPU_SHADER_CALCULATOR_H_

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <vector>

#include "mediapipe/framework/api3/any.h"
#include "mediapipe/framework/api3/contract.h"
#include "mediapipe/framework/api3/node.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/webgpu/webgpu_shader_calculator.pb.h"
#include "mediapipe/gpu/webgpu/webgpu_texture_buffer_3d.h"
#include "mediapipe/gpu/webgpu/webgpu_texture_view.h"

namespace mediapipe::api3 {

// Compiles a given WGSL shader, and runs it over the input WebGPU-backed
// GpuBuffer streams to produce an output WebGPU-backed GpuBuffer stream.
//
// - We expect a "Params" struct in the shader for our uniforms.
// - We will automatically pipe in values for 'outputSize' and 'time' using the
//   size of the output texture and the timestamp in seconds, respectively.
// - Otherwise, all uniforms in Params are expected to be f32 or vectors of f32.
// - We will bind all f32 uniforms to INPUT_FLOAT streams, matching the order
//   those streams are given to the order of f32 uniforms in the Params struct.
// - And we will bind all vec2<f32>, vec3<f32>, and vec4<f32> uniforms to
//   INPUT_FLOAT_VEC streams, matching the order those streams are given to the
//   order of vec*<f32> uniforms in the Params struct.
// - We bind all input buffers, matching the order they are given to the
//   calculator via INPUT_BUFFER, with the order they are listed in the shader
//   source code.
// - We similarly bind all input 3d buffers (if any), matching the order they
//   are given to the calculator via INPUT_BUFFER_3D, with the order they are
//   listed in the shader source code.
struct WebGpuShaderNode : Node<"WebGpuShaderCalculator"> {
  template <typename S>
  struct Contract {
    // ***  INPUTS  ***

    // List of input buffers. Must contain one for every 2d texture the shader
    // code references.
    Repeated<Input<S, GpuBuffer>> input_buffers{"INPUT_BUFFER"};
    // List of 3d input buffers, for compute shaders. Must contain one for
    // every 3d texture the shader code references.
    Repeated<Input<S, WebGpuTextureBuffer3d>> input_buffers_3d{
        "INPUT_BUFFER_3D"};
    // List of float value streams. Must contain one for every float uniform
    // the shader code references.
    Repeated<Input<S, float>> input_floats{"INPUT_FLOAT"};
    // List of float vector streams. Must contain one for every vec2, vec3, or
    // vec4 uniform the shader code references.
    Repeated<Input<S, std::vector<float>>> input_float_vecs{"INPUT_FLOAT_VEC"};
    // Input stream which will dynamically set the rendering output width.
    // Overrides other methods of setting this property.
    Optional<Input<S, int32_t>> width{"WIDTH"};
    // Input stream which will dynamically set the rendering output height.
    // Overrides other methods of setting this property.
    Optional<Input<S, int32_t>> height{"HEIGHT"};
    // Input stream which will dynamically set the rendering output depth.
    // This is unused for normal (2d) rendering, and if used will change the
    // output type to be a WebGpuTextureBuffer3d. Overrides other methods of
    // setting this property.
    Optional<Input<S, int32_t>> depth{"DEPTH"};
    // Stream which is used (in the absence of INPUT_BUFFER and INPUT_FLOAT
    // streams) to trigger output of an input-free shader.
    Optional<Input<S, Any>> trigger{"TRIGGER"};

    // ***  OUTPUTS  ***

    // Frames containing the result of the 2D rendering. This will be the
    // output stream unless 3D compute shading is occurring.
    Optional<Output<S, GpuBuffer>> output{"OUTPUT"};
    // Frames containing the result of the 3D compute shading, when an output
    // depth has been specified.
    Optional<Output<S, WebGpuTextureBuffer3d>> output_3d{"OUTPUT_3D"};

    // ***  OPTIONS  ***

    Options<S, mediapipe::WebGpuShaderCalculatorOptions> options;
  };
};

}  // namespace mediapipe::api3

#endif  // MEDIAPIPE_GPU_WEBGPU_WEBGPU_SHADER_CALCULATOR_H_

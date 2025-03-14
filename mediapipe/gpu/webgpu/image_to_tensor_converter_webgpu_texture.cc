#include "mediapipe/gpu/webgpu/image_to_tensor_converter_webgpu_texture.h"

#include <webgpu/webgpu_cpp.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "mediapipe/calculators/tensor/image_to_tensor_converter.h"
#include "mediapipe/calculators/tensor/image_to_tensor_utils.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/gpu/webgpu/webgpu_service.h"
#include "mediapipe/gpu/webgpu/webgpu_texture_view.h"
#include "mediapipe/gpu/webgpu/webgpu_utils.h"

namespace mediapipe {
namespace {

constexpr uint32_t kTileSize = 8;

// Similar to GetTransposedRotatedSubRectToRectTransformMatrix, however it is
// modified to be optimal with WebGPU.
//
//  - Output is a 3x3 matrix instead of 4x4. (Padded to 3x4 for WebGPU.)
//  - Input coordinates are pixels in output rather than [0, 1].
//  - Output coordinates are pixels in input rather than [0, 1].
//  - Unused "flip_horizontally" matrix removed.
void GetTransposedRotatedSubRectToRectTransformMatrixWebGpu(
    const RotatedRect& sub_rect, int output_width, int output_height,
    std::array<float, 12>& matrix) {
  // The resulting matrix is multiplication of below commented out matrices:
  //   translate_matrix
  //     * rotate_matrix
  //     * initial_translate_matrix
  //     * scale_matrix

  const float a = sub_rect.width / static_cast<float>(output_width);
  const float b = sub_rect.height / static_cast<float>(output_height);
  // Matrix to convert X,Y from [0, output_size] to [0, sub_rect_size] range
  // "scale_matrix"
  //
  // { a   ,  0.0f,  0.0f}
  // { 0.0f,  b   ,  0.0f}
  // { 0.0f,  0.0f,  1.0f}

  const float c = sub_rect.width / 2.0f;
  const float d = sub_rect.height / 2.0f;
  // Matrix to convert X,Y to [-sub_rect_size / 2, sub_rect_size / 2] range
  // "initial_translate_matrix"
  //
  // { 1.0f,  0.0f, -c   }
  // { 0.0f,  1.0f, -d   }
  // { 0.0f,  0.0f,  1.0f}

  const float e = std::cos(sub_rect.rotation);
  const float f = std::sin(sub_rect.rotation);
  // Matrix to do rotation around Z axis "rotate_matrix"
  //
  // {    e,   -f, 0.0f}
  // {    f,    e, 0.0f}
  // { 0.0f, 0.0f, 1.0f}

  const float g = sub_rect.center_x;
  const float h = sub_rect.center_y;
  // Matrix to do X,Y translation of sub rect within parent rect
  // "translate_matrix"
  //
  // {1.0f, 0.0f, g   }
  // {0.0f, 1.0f, h   }
  // {0.0f, 0.0f, 1.0f}

  // Note: Each column is 4 elements long because WebGPU adds padding.

  // column 1
  matrix[0] = a * e;
  matrix[1] = a * f;
  matrix[2] = 0.0f;
  matrix[3] = 0.0f;

  // column 2
  matrix[4] = -b * f;
  matrix[5] = b * e;
  matrix[6] = 0.0f;
  matrix[7] = 0.0f;

  // column 3
  matrix[8] = d * f + g - c * e;
  matrix[9] = -c * f + h - d * e;
  matrix[10] = 1.0f;
  matrix[11] = 0.0f;
}

// Crude ImageToTensorConverter that does the minimal for WebGPU textures.
class Converter : public ImageToTensorConverter {
 public:
  // TODO: Support `input_starts_at_bottom` and `border_mode`.
  Converter(CalculatorContext* cc)
      : service_(cc->Service(kWebGpuService).GetObject()) {
    const std::string shader = absl::StrFormat(R"(
struct Parameters {
  transform: mat3x3f,
  output_size : vec2<u32>,
  value_transform : vec2<f32>,
};

@group(0) @binding(0) var input : texture_2d<f32>;
@group(0) @binding(1) var output : texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params : Parameters;

@compute @workgroup_size(%d, %d)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= params.output_size.x || gid.y >= params.output_size.y) {
    return;
  }
  let input_coord = (params.transform * vec3<f32>(vec2<f32>(gid.xy), 1.0f)).xy;
  let input_value = textureLoad(input, vec2<i32>(input_coord), 0);
  let output_value = params.value_transform.x * input_value.xyz
      + vec3<f32>(params.value_transform.y);
  textureStore(
      output, vec2<i32>(gid.xy),
      vec4<f32>(output_value, 1.0));
}
)",
                                               kTileSize, kTileSize);

    // Create the shader module.
    wgpu::ShaderSourceWGSL wgsl;
    wgsl.code = shader.c_str();
    wgpu::ShaderModuleDescriptor shader_desc = {.nextInChain = &wgsl};
    wgpu::ShaderModule module =
        service_.device().CreateShaderModule(&shader_desc);

    // Create the compute pipeline.
    wgpu::ComputePipelineDescriptor pipeline_desc = {
        .compute =
            {
                .module = module,
                .entryPoint = "main",
                .constantCount = 0,
                .constants = nullptr,
            },
    };
    pipeline_ =
        WebGpuCreateComputePipelineAsync(service_.device(), &pipeline_desc);

    // Create a uniform buffer for the parameters.
    wgpu::BufferDescriptor buffer_desc = {
        .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
        .size = sizeof(Parameters),
    };
    params_buffer_ = service_.device().CreateBuffer(&buffer_desc);
  }

  absl::Status Convert(const mediapipe::Image& input, const RotatedRect& roi,
                       float range_min, float range_max,
                       int tensor_buffer_offset, Tensor& output_tensor) final {
    RET_CHECK_EQ(tensor_buffer_offset, 0)
        << "The non-zero tensor_buffer_offset input is not supported yet.";
    const auto& output_shape = output_tensor.shape();
    MP_RETURN_IF_ERROR(ValidateTensorShape(output_shape));

    const int output_height = output_shape.dims[1];
    const int output_width = output_shape.dims[2];
    const wgpu::Device& device = service_.device();

    // TODO: Find out why this workaround is needed.
    // auto image_view = input.GetGpuBuffer().GetReadView<WebGpuTextureView>();
    const GpuBuffer& in_buffer = input.GetGpuBuffer(/*upload_to_gpu=*/false);
    WebGpuTextureView image_view = in_buffer.GetReadView<WebGpuTextureView>();
    wgpu::Texture src_texture = image_view.texture();

    auto tensor_view = output_tensor.GetWebGpuTexture2dWriteView(service_);
    wgpu::Texture dst_texture = tensor_view.name();

    // Update shader parameters if the size has changed.
    if (output_width == 0 || output_height == 0) {
      return absl::InternalError("Empty output dimensions.");
    }
    Parameters params;
    GetTransposedRotatedSubRectToRectTransformMatrixWebGpu(
        roi, output_width, output_height, params.transform_matrix);
    params.output_width = output_width;
    params.output_height = output_height;
    MP_ASSIGN_OR_RETURN(auto transform, GetValueRangeTransformation(
                                            0.f, 1.f, range_min, range_max));
    params.value_transform_scale = transform.scale;
    params.value_transform_offset = transform.offset;

    if (memcmp(&params, &last_params_, sizeof(Parameters)) != 0) {
      last_params_ = params;
      device.GetQueue().WriteBuffer(params_buffer_, 0, &last_params_,
                                    sizeof(Parameters));
    }

    // Create the bind group.
    // [[group(0), binding(0)]] is the input texture.
    // [[group(0), binding(1)]] is the output texture.
    // [[group(0), binding(2)]] is the shader parameters uniform buffer.
    wgpu::BindGroupEntry entries[] = {
        {
            .binding = 0,
            .textureView = src_texture.CreateView(),
        },
        {
            .binding = 1,
            .textureView = dst_texture.CreateView(),
        },
        {
            .binding = 2,
            .buffer = params_buffer_,
            .size = sizeof(Parameters),
        },
    };
    MP_ASSIGN_OR_RETURN(wgpu::ComputePipeline * pipeline, pipeline_.Get());
    wgpu::BindGroupDescriptor bind_group_desc = {
        .layout = pipeline->GetBindGroupLayout(0),
        .entryCount = sizeof(entries) / sizeof(entries[0]),
        .entries = entries,
    };
    wgpu::BindGroup bind_group = device.CreateBindGroup(&bind_group_desc);

    // Round up the number of workgroups to cover the whole texture.
    const uint32_t num_groups_x = (output_width + kTileSize - 1) / kTileSize;
    const uint32_t num_groups_y = (output_height + kTileSize - 1) / kTileSize;

    // Create and submit a command buffer that dispatches the compute shader.
    auto command_encoder = device.CreateCommandEncoder();
    auto pass_encoder = command_encoder.BeginComputePass();
    pass_encoder.SetPipeline(*pipeline);
    pass_encoder.SetBindGroup(0, bind_group);
    pass_encoder.DispatchWorkgroups(num_groups_x, num_groups_y);
    pass_encoder.End();
    wgpu::CommandBuffer command_buffers[] = {
        command_encoder.Finish(),
    };
    device.GetQueue().Submit(std::size(command_buffers), command_buffers);

    return absl::OkStatus();
  }

 private:
  absl::Status ValidateTensorShape(const Tensor::Shape& output_shape) {
    RET_CHECK_EQ(output_shape.dims.size(), 4)
        << "Wrong output dims size: " << output_shape.dims.size();
    RET_CHECK_EQ(output_shape.dims[0], 1)
        << "Handling batch dimension not equal to 1 is not implemented in this "
           "converter.";
    RET_CHECK_EQ(output_shape.dims[3], 3)
        << "Wrong output channel: " << output_shape.dims[3];
    return absl::OkStatus();
  }

  const WebGpuService& service_;
  WebGpuAsyncFuture<wgpu::ComputePipeline> pipeline_;
  wgpu::Buffer params_buffer_;

  struct Parameters {  // Must match `Parameters` in WGSL above.
    std::array<float, 12> transform_matrix;
    uint32_t output_width;
    uint32_t output_height;
    float value_transform_scale;
    float value_transform_offset;
  };
  Parameters last_params_ = {
      .output_width = 0,
      .output_height = 0,
      .value_transform_scale = 0.0,
      .value_transform_offset = 0.0,
  };
};

}  // namespace

absl::StatusOr<std::unique_ptr<ImageToTensorConverter>>
CreateImageToWebGpuTextureTensorConverter(CalculatorContext* cc) {
  return std::make_unique<Converter>(cc);
}

}  // namespace mediapipe

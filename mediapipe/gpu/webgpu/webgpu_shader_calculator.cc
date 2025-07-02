#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/deps/re2.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/resources.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/gpu/webgpu/webgpu_service.h"
#include "mediapipe/gpu/webgpu/webgpu_shader_calculator.pb.h"
#include "mediapipe/gpu/webgpu/webgpu_texture_buffer_3d.h"
#include "mediapipe/gpu/webgpu/webgpu_texture_view.h"
#include "mediapipe/gpu/webgpu/webgpu_utils.h"

namespace mediapipe {

// Compiles a given WGSL shader, and runs it over the input WebGPU-backed
// GpuBuffer streams to produce an output WebGPU-backed GpuBuffer stream.
// We expect a "Params" struct in the shader for our uniforms. We will
// automatically pipe in values for 'outputSize' and 'time' using the size of
// the output texture and the timestamp in seconds, respectively. Otherwise, all
// uniforms in Params are expected to be f32 or vectors of f32. We will bind all
// f32 uniforms to INPUT_FLOAT streams, matching the order those streams are
// are given to the order of f32 uniforms in the Params struct. And we will
// bind all vec2<f32>, vec3<f32>, and vec4<f32> uniforms to INPUT_FLOAT_VEC
// streams, matching the order those streams are given to the order of vec*<f32>
// uniforms in the Params struct.
// We bind all input buffers, matching the order they are given to the
// calculator via INPUT_BUFFER, with the order they are listed in the shader
// source code. We similarly bind all input 3d buffers (if any), matching the
// order they are given to the calculator via INPUT_BUFFER_3D, with the order
// they are listed in the shader source code.
//
// Inputs:
//   TRIGGER (Any):
//     Stream which is used (in the absence of INPUT_BUFFER and INPUT_FLOAT
//     streams) to trigger output of an input-free shader.
//   INPUT_BUFFER (GpuBuffer, repeated):
//     List of input buffers. Must contain one for every 2d texture the shader
//     code references.
//   INPUT_BUFFER_3D (WebGpuTextureBuffer3d, repeated):
//     List of 3d input buffers, for compute shaders. Must contain one for every
//     3d texture the shader code references.
//   INPUT_FLOAT (float, repeated):
//     List of float value streams. Must contain one for every float uniform
//     the shader code references.
//   INPUT_FLOAT_VEC (vec<float>, repeated):
//     List of float vector streams. Must contain one for every vec2, vec3, or
//     vec4 uniform the shader code references.
//   WIDTH (int32_t):
//     Input stream which will dynamically set the rendering output width.
//     Overrides other methods of setting this property.
//   HEIGHT (int32_t):
//     Input stream which will dynamically set the rendering output height.
//     Overrides other methods of setting this property.
//   DEPTH (int32_t):
//     Input stream which will dynamically set the rendering output depth.
//     This is unused for normal (2d) rendering, and if used will change the
//     output type to be a WebGpuTextureBuffer3d. Overrides other methods of
//     setting this property.
//
// Outputs:
//   OUTPUT (GpuBuffer)
//     Frames containing the result of the 2D rendering. This will be the output
//     stream unless 3D compute shading is occurring.
//   OUTPUT_3D (WebGpuTextureBuffer3d)
//     Frames containing the result of the 3D compute shading, when an output
//     depth has been specified.

namespace {

using ::mediapipe::api2::AnyType;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::NodeImpl;
using ::mediapipe::api2::NodeIntf;
using ::mediapipe::api2::Output;

// One query for start time, one for end time.
constexpr uint32_t kQueryBufferByteSize = 2 * sizeof(uint64_t);

// WebGpu uses this as a default for each dimension.
constexpr size_t kDefaultWorkgroupSize = 1;

// WebGpu imposes a minimum buffer size for queries, so we may need to pad.
constexpr uint32_t kMinQueryBufferSize = 256u;

// Search terms we use for parsing shader code.
constexpr char kParseTermBinding[] = "@binding(";
constexpr char kParseTermParams[] = "struct Params {";
constexpr char kParseTermWorkgroup[] = "@workgroup_size(";
const size_t kParseTermBindingSize = std::strlen(kParseTermBinding);
const size_t kParseTermParamsSize = std::strlen(kParseTermParams);
const size_t kParseTermWorkgroupSize = std::strlen(kParseTermWorkgroup);

// If no shader provided, we assume passthrough with same-size input and output.
constexpr char kDefaultWebGpuShaderSource[] = R"(
struct Params {
  outputSize : vec2<i32>
}

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var outputTex : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(8, 8)
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>
) {
  let outputCoord = vec2<i32>(gid.xy);
  if (outputCoord.x >= params.outputSize.x
      || outputCoord.y >= params.outputSize.y) {
    return;
  }
  let input = textureLoad(inputTex, outputCoord, 0);
  textureStore(outputTex, outputCoord, input);
}
)";

#ifdef __EMSCRIPTEN__
// Expose the information which is used in JavaScript later to display the
// WebGPU metrics in real-time.
EM_JS(void, ExposeProfilingResults,
      (const char* calc_name, void* wgpu_buffer_ptr, int32_t num_repetitions), {
        // Initialize profiling call
        if (!Module.WEBGPU_SHADER_CALC_PROFILE) {
          Module.WEBGPU_SHADER_CALC_PROFILE = async() => {
            console.log('WebGpuShaderCalculator profiling results: ');
            for (const calc in Module.WEBGPU_SHADER_CALC_METRICS) {
              const readBuffer = Module.WEBGPU_SHADER_CALC_METRICS[calc].buffer;
              await readBuffer.mapAsync(GPUMapMode.READ);
              const resultsBuffer =
                  new BigInt64Array(readBuffer.getMappedRange());
              const timeDelta = resultsBuffer[1] - resultsBuffer[0];
              if (Number.MIN_SAFE_INTEGER < timeDelta &&
                  timeDelta < Number.MAX_SAFE_INTEGER) {
                const timeMs = Number(timeDelta) / 1000000;
                const numRepetitions =
                    Module.WEBGPU_SHADER_CALC_METRICS[calc].repetitions;
                console.log(`${calc} : ${timeMs / numRepetitions} ms`);
              }
              readBuffer.unmap();
            }
            console.log('Finished printing WebGpuShaderCalculator profiling.');
          };
        }

        // Expose the buffer directly to our table.
        const gpuReadBuffer = WebGPU.getJsObject(wgpu_buffer_ptr);
        Module.WEBGPU_SHADER_CALC_METRICS =
            Module.WEBGPU_SHADER_CALC_METRICS || {};
        Module.WEBGPU_SHADER_CALC_METRICS[UTF8ToString(calc_name)] = {
          buffer : gpuReadBuffer,
          repetitions : num_repetitions,
        };
      });
#else
void ExposeProfilingResults(const char* calc_name, void* wgpu_buffer_ptr,
                            int32_t num_repetitions) {
  ABSL_LOG(WARNING) << "Exposing profiling results only implemented on web.";
}
#endif  // __EMSCRIPTEN__

// Quick helper to remove comments in shader code before parsing it for various
// search terms and tokens.
std::string RemoveComments(const std::string& str) {
  // First we remove all `/* ... */` blocks.
  // We match the sequence of "/*", any non-null character sequence (so we can
  // include multi-line comment blocks), and finally "*/". We use '?' to request
  // non-greedy matching.
  std::string new_str = str;
  RE2 pattern_comment_block("/\\*[^\\0]*?\\*/");
  RE2::GlobalReplace(&new_str, pattern_comment_block, "");

  // Then we remove all remaining `// ...` lines.
  // For this, we want greedy matching up to a newline.
  RE2 pattern_comment_line("//.*");
  RE2::GlobalReplace(&new_str, pattern_comment_line, "");
  return new_str;
}

class ScopedWebGpuErrorHandler {
 public:
  ScopedWebGpuErrorHandler(
      WebGpuService* service, std::string_view callsite,
      mediapipe::Timestamp timestamp = mediapipe::Timestamp::Unset())
      : service_(service), callsite_(callsite), timestamp_(timestamp) {
    PushErrorScopes();
  }

  ~ScopedWebGpuErrorHandler() { PopErrorScopes(); }

 private:
  WebGpuService* service_ = nullptr;
  std::string callsite_;
  mediapipe::Timestamp timestamp_;

  void PushErrorScopes();
  void PopErrorScopes();

  static std::string MapErrorTypesToString(wgpu::ErrorType type) {
    switch (type) {
      case wgpu::ErrorType::NoError:
        return "NoError";
      case wgpu::ErrorType::Validation:
        return "Validation";
      case wgpu::ErrorType::OutOfMemory:
        return "OutOfMemory";
      case wgpu::ErrorType::Internal:
        return "Internal";
      default:
        return "Unknown";
    }
  }
};

void ScopedWebGpuErrorHandler::PushErrorScopes() {
  service_->device().PushErrorScope(wgpu::ErrorFilter::Validation);
  service_->device().PushErrorScope(wgpu::ErrorFilter::OutOfMemory);
  service_->device().PushErrorScope(wgpu::ErrorFilter::Internal);
}

void ScopedWebGpuErrorHandler::PopErrorScopes() {
  std::function<void(wgpu::PopErrorScopeStatus, wgpu::ErrorType,
                     wgpu::StringView)>
      callback = [callsite = callsite_, timestamp = timestamp_](
                     wgpu::PopErrorScopeStatus status, wgpu::ErrorType type,
                     wgpu::StringView message) {
        if (type == wgpu::ErrorType::NoError) {
          return;
        }
        std::string timestamp_str =
            timestamp.IsSpecialValue()
                ? timestamp.DebugString()
                : absl::StrFormat("%d", timestamp.Value());

        // Note that we only log the error message here and do not bubble up
        // an error status. Given the asynchronous nature of WebGPU and its
        // error handling via callbacks, we would only be able to return
        // errors in a Process() call after a Process() call in which the error
        // occurred.
        ABSL_LOG(ERROR) << "WebGPU error of type: "
                        << MapErrorTypesToString(type) << " encountered in "
                        << callsite << " at timestamp: " << timestamp_str
                        << ". "
                        << " Error message: " << std::string(message);
      };

  // We pushed 3 error scopes, so we need to pop 3.
  service_->device().PopErrorScope(wgpu::CallbackMode::AllowSpontaneous,
                                   callback);
  service_->device().PopErrorScope(wgpu::CallbackMode::AllowSpontaneous,
                                   callback);
  service_->device().PopErrorScope(wgpu::CallbackMode::AllowSpontaneous,
                                   callback);
}

// Quick helper to remove whitespace and parse our "a : b," list into tokens.
std::string ExtractParamFromTo(const std::string& str, int start_index,
                               int end_index) {
  std::string token = "";
  for (int i = start_index; i <= end_index; i++) {
    if (str[i] == ',' || str[i] == '}' || str[i] == ';' || str[i] == ':') break;
    if (str[i] == ' ' || str[i] == '\n') continue;
    token += str[i];
  }
  return token;
}

struct ParamOffsets {
  int num_params = 0;
  std::optional<int> output_size_offset;
  std::optional<int> time_offset;
  std::vector<int> float_offsets;
  std::vector<int> float_vec_offsets;
};

absl::StatusOr<ParamOffsets> GetParamOffsets(const std::string& source) {
  ParamOffsets offsets;

  // First we extract the Params struct info.
  int param_start = source.find(kParseTermParams);
  if (param_start == std::string::npos) {
    return absl::InternalError(
        "Could not parse Params struct from WebGPU shader.");
  }
  param_start += kParseTermParamsSize;  // skip over search term
  int param_end = source.find('}', param_start);
  std::string param_source =
      source.substr(param_start, param_end - param_start);

  param_start = 0;
  int offset = 0;
  while (true) {
    int param_split = param_source.find(':', param_start);
    if (param_split == std::string::npos) {
      offsets.num_params = offset;
      return offsets;
    }

    std::string param_name =
        ExtractParamFromTo(param_source, param_start, param_split);

    int param_end = param_source.find(',', param_split);
    if (param_end == std::string::npos) param_end = param_source.size() - 1;

    std::string param_type =
        ExtractParamFromTo(param_source, param_split + 1, param_end);

    // Now we handle parameters we wish to auto-hook up:
    //   outputSize and time
    // And otherwise, we assume parameters will come from input streams
    if (param_name == "outputSize") {
      offsets.output_size_offset = offset;
      offset += 3;  // we just always use 3, in case of depth
    } else if (param_name == "time") {
      offsets.time_offset = offset;
      offset += 1;
    } else if (param_type == "f32") {
      offsets.float_offsets.push_back(offset);
      offset += 1;
    } else if (param_type == "vec2<f32>") {
      offsets.float_vec_offsets.push_back(offset);
      offset += 2;
    } else if (param_type == "vec3<f32>") {
      offsets.float_vec_offsets.push_back(offset);
      offset += 3;
    } else if (param_type == "vec4<f32>") {
      offsets.float_vec_offsets.push_back(offset);
      offset += 4;
    } else {
      return absl::InternalError(
          absl::StrCat("Cannot parse Params type: ", param_type, " for ",
                       "parameter ", param_name));
    }

    param_start = param_end + 1;
  }
}

// We return a vector of all binding locations which match the given term.
absl::StatusOr<std::vector<int>> GetBindingLocations(
    const std::string& search_term, const std::string& source) {
  std::vector<int> binding_locations;
  std::string binding_str;
  int next_binding_start = 0;
  int next_binding_end;
  while (true) {
    next_binding_start = source.find(kParseTermBinding, next_binding_start);
    if (next_binding_start == std::string::npos) break;
    next_binding_start += kParseTermBindingSize;  // skip over search term
    next_binding_end = source.find(';', next_binding_start);
    binding_str = source.substr(next_binding_start,
                                next_binding_end - next_binding_start);
    if (binding_str.find(search_term) != std::string::npos) {
      int binding_str_end = binding_str.find(')');
      if (binding_str_end != std::string::npos) {
        int location = std::stoi(binding_str.substr(0, binding_str_end));
        binding_locations.push_back(location);
      } else {
        return absl::InternalError(
            absl::StrCat("Binding could not be parsed at: ", binding_str));
      }
    }
  }
  return binding_locations;
}

// We expect a unique location for these, or else none at all, in which case we
// return std::nullopt.
absl::StatusOr<std::optional<int>> GetBindingLocation(
    const std::string& search_term, const std::string& source) {
  MP_ASSIGN_OR_RETURN(const auto& locations,
                      GetBindingLocations(search_term, source));
  if (locations.size() > 1) {
    return absl::InternalError(absl::StrFormat(
        "Expected a unique binding location for %s, but found %d.", search_term,
        locations.size()));
  }
  if (locations.empty()) return std::nullopt;
  return locations[0];
}

// We return a vector of the workgroup sizes.
std::vector<int> GetWorkgroupSizes(const std::string& source) {
  std::vector<int> results;
  int expr_start = source.find(kParseTermWorkgroup);
  if (expr_start == std::string::npos) return results;
  expr_start += kParseTermWorkgroupSize;  // skip over search term
  int expr_end = source.find(')', expr_start);
  std::string expr_str = source.substr(expr_start, expr_end - expr_start);
  std::vector<absl::string_view> terms = absl::StrSplit(expr_str, ',');
  for (const auto& term : terms) {
    int new_val;
    if (absl::SimpleAtoi(term, &new_val)) {
      results.push_back(new_val);
    } else {
      ABSL_LOG(WARNING) << "Error parsing workgroup size at: " << term << " in "
                        << expr_str;
    }
  }
  return results;
}

// We return the output format for the given texture type. Specifically, we
// assume there is only one output in the shader which is formatted as:
// "[TEXTURE_TYPE]<[FORMAT], write>", and we want to return [FORMAT].
std::string GetOutputFormat(const std::string& texture_type,
                            const std::string& source) {
  // Find texture_type in our shader code
  const int expr_start = source.find(texture_type);
  if (expr_start == std::string::npos) {
    return "Error parsing output format: cannot find " + texture_type;
  }

  // Then find the next '<' character.
  int term_start = source.find('<', expr_start);
  if (term_start == std::string::npos) {
    return "Error parsing output format: cannot find starting '<'";
  }
  term_start++;

  // We'll also find the enclosing '>' character, just so we can provide a more
  // helpful error message if/when the term isn't formatted properly.
  const int term_end = source.find('>', term_start);
  if (term_end == std::string::npos) {
    return "Error parsing output format: cannot find ending '>'";
  }

  // Then parse start of term until following ',' character.
  const int comma = source.find(',', term_start);
  if (comma == std::string::npos || comma > term_end) {
    return "Error parsing output format: not of type '<FORMAT, write>'";
  }

  // Return that substring if it exists.
  return source.substr(term_start, comma - term_start);
}

}  // namespace

class WebGpuShaderCalculator
    : public NodeImpl<NodeIntf, WebGpuShaderCalculator> {
 public:
  WebGpuShaderCalculator() = default;
  WebGpuShaderCalculator(const WebGpuShaderCalculator&) = delete;
  ~WebGpuShaderCalculator() override = default;

  static constexpr auto kCalculatorName = "WebGpuShaderCalculator";

  static constexpr Input<GpuBuffer>::Multiple kInputBuffers{"INPUT_BUFFER"};
  static constexpr Input<WebGpuTextureBuffer3d>::Multiple kInputBuffers3d{
      "INPUT_BUFFER_3D"};
  static constexpr Input<float>::Multiple kInputFloats{"INPUT_FLOAT"};
  static constexpr Input<std::vector<float>>::Multiple kInputFloatVecs{
      "INPUT_FLOAT_VEC"};
  static constexpr Input<int32_t>::Optional kInputWidth{"WIDTH"};
  static constexpr Input<int32_t>::Optional kInputHeight{"HEIGHT"};
  static constexpr Input<int32_t>::Optional kInputDepth{"DEPTH"};
  static constexpr Input<AnyType>::Optional kInputTrigger{"TRIGGER"};
  static constexpr Output<GpuBuffer>::Optional kOutput{"OUTPUT"};
  static constexpr Output<WebGpuTextureBuffer3d>::Optional kOutput3d{
      "OUTPUT_3D"};

  MEDIAPIPE_NODE_CONTRACT(kInputBuffers, kInputBuffers3d, kInputFloats,
                          kInputFloatVecs, kInputWidth, kInputHeight,
                          kInputDepth, kInputTrigger, kOutput, kOutput3d);

  static absl::Status UpdateContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status InitWebGpuShader();
  void InitProfiling();
  absl::Status WebGpuBindAndRender(
      CalculatorContext* cc, const wgpu::ComputePipeline& pipeline, int width,
      int height, int depth, const std::vector<WebGpuTextureView>& src_textures,
      const std::vector<WebGpuTextureView>& src_textures_3d,
      const std::vector<float>& src_floats,
      const std::vector<std::vector<float>>& src_float_vecs);
  absl::Status WebGpuBindAndRenderToView(
      CalculatorContext* cc, const wgpu::ComputePipeline& pipeline, int width,
      int height, int depth, const std::vector<WebGpuTextureView>& src_textures,
      const std::vector<WebGpuTextureView>& src_textures_3d,
      const std::vector<float>& src_floats,
      const std::vector<std::vector<float>>& src_float_vecs,
      WebGpuTextureView& out_view);
  void HandleEmptyPacket(CalculatorContext* cc, int index,
                         bool has_gpu_buffer_input,
                         const std::string& stream_debug_name);

  std::string shader_source_;

  // These will be grabbed from @workgroup_size in the shader, if available.
  size_t workgroup_size_x_ = kDefaultWorkgroupSize;
  size_t workgroup_size_y_ = kDefaultWorkgroupSize;
  size_t workgroup_size_z_ = kDefaultWorkgroupSize;

  GpuBufferFormat output_format_ = GpuBufferFormat::kRGBA32;
  std::optional<int32_t> output_width_;
  std::optional<int32_t> output_height_;
  std::optional<int32_t> output_depth_;

  std::optional<int> sampler_binding_;
  std::optional<int> output_texture_binding_;
  std::optional<int> uniform_binding_;
  std::vector<int> input_texture_bindings_;
  std::vector<int> input_texture_3d_bindings_;

  ParamOffsets param_offsets_;
  uint32_t params_size_;
  std::unique_ptr<float[]> params_data_;

  bool passthrough_first_buffer_on_empty_packets_ = true;

  WebGpuService* service_ = nullptr;
  WebGpuAsyncFuture<wgpu::ComputePipeline> pipeline_future_;
  wgpu::Buffer params_;
  wgpu::Sampler sampler_;

  // For profiling (requires extensions and Chrome Canary)
  bool profile_ = false;
  uint32_t repetitions_ = 1000;
  uint32_t skip_starting_frames_ = 100;
  wgpu::QuerySet query_set_;
  wgpu::Buffer query_buffer_;
  wgpu::Buffer dst_buffer_;
  wgpu::PassTimestampWrites timestamp_writes_;
};

absl::Status WebGpuShaderCalculator::UpdateContract(
    mediapipe::CalculatorContract* cc) {
  RET_CHECK(kOutput(cc).IsConnected() || kOutput3d(cc).IsConnected())
      << "Output tag expected.";
  RET_CHECK(kOutput(cc).IsConnected() != kOutput3d(cc).IsConnected())
      << "Only one output tag expected.";
  RET_CHECK(kInputBuffers(cc).Count() > 0 || kInputBuffers3d(cc).Count() > 0 ||
            kInputTrigger(cc).IsConnected())
      << "At least one input tag expected.";
  cc->UseService(mediapipe::kWebGpuService);
  return absl::OkStatus();
}

absl::Status WebGpuShaderCalculator::Open(CalculatorContext* cc) {
  // Grab our shader sources from options, or default init them.
  const mediapipe::WebGpuShaderCalculatorOptions& options =
      cc->Options().GetExtension(mediapipe::WebGpuShaderCalculatorOptions::ext);

  if (options.has_shader_path()) {
    std::unique_ptr<Resource> resource_shader_source;
    MP_ASSIGN_OR_RETURN(resource_shader_source,
                        cc->GetResources().Get(options.shader_path()));
    shader_source_ = resource_shader_source->ToStringView();
  } else if (options.has_shader_source()) {
    shader_source_ = options.shader_source();
  } else {
    shader_source_ = kDefaultWebGpuShaderSource;
  }

  if (options.has_output_width()) {
    output_width_ = options.output_width();
  }
  if (options.has_output_height()) {
    output_height_ = options.output_height();
  }
  if (options.has_output_depth()) {
    output_depth_ = options.output_depth();
  }

  if (options.has_profiling_options()) {
    const auto profiling_options = options.profiling_options();
    if (profiling_options.has_enable()) {
      profile_ = profiling_options.enable();
    }
    if (profiling_options.has_repetitions()) {
      repetitions_ = profiling_options.repetitions();
    }
    if (profiling_options.has_skip_starting_frames()) {
      skip_starting_frames_ = profiling_options.skip_starting_frames();
    }
  }

  // Request WebGpu resources
  service_ = &cc->Service(kWebGpuService).GetObject();
  MP_RETURN_IF_ERROR(InitWebGpuShader());
  if (profile_) InitProfiling();

  return absl::OkStatus();
}

void WebGpuShaderCalculator::InitProfiling() {
  // TODO: Check for timestamp extension, so we can error out
  // gracefully if user is running profiling in the wrong environment/setup.

  ScopedWebGpuErrorHandler error_handler(
      service_, "WebGpuShaderCalculator::InitProfiling");

  // Create buffers: one for the queries, and one for exposed results.
  // We currently have only 2 queries in our set: start and end.
  wgpu::BufferDescriptor query_buffer_desc = {
      .usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::QueryResolve,
      .size = std::max(kMinQueryBufferSize, kQueryBufferByteSize),
  };
  query_buffer_ = service_->device().CreateBuffer(&query_buffer_desc);
  wgpu::BufferDescriptor dst_buffer_desc = {
      .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
      .size = std::max(kMinQueryBufferSize, kQueryBufferByteSize),
  };
  dst_buffer_ = service_->device().CreateBuffer(&dst_buffer_desc);

  // Then make our query set
  wgpu::QuerySetDescriptor query_set_descriptor = {
      .type = wgpu::QueryType::Timestamp,
      .count = 2,
  };
  query_set_ = service_->device().CreateQuerySet(&query_set_descriptor);

  // Set our timestamp_writes_ data for quick compute pass descriptor production
  timestamp_writes_ = {.querySet = query_set_,
                       .beginningOfPassWriteIndex = 0,
                       .endOfPassWriteIndex = 1};
}

absl::Status WebGpuShaderCalculator::InitWebGpuShader() {
  ScopedWebGpuErrorHandler error_handler(
      service_, "WebGpuShaderCalculator::InitWebGpuShader");
  const std::string comment_free_shader_src = RemoveComments(shader_source_);

  // Parse shader to grab workgroup sizes, if overridden
  const auto workgroup_sizes = GetWorkgroupSizes(comment_free_shader_src);
  if (!workgroup_sizes.empty()) workgroup_size_x_ = workgroup_sizes[0];
  if (workgroup_sizes.size() > 1) workgroup_size_y_ = workgroup_sizes[1];
  if (workgroup_sizes.size() > 2) workgroup_size_z_ = workgroup_sizes[2];

  // Parse shader to grab binding locations. Try 2d first, then 3d.
  // TODO: Allow for multiple outputs.
  MP_ASSIGN_OR_RETURN(
      output_texture_binding_,
      GetBindingLocation("texture_storage_2d", comment_free_shader_src));
  if (!output_texture_binding_) {
    MP_ASSIGN_OR_RETURN(
        output_texture_binding_,
        GetBindingLocation("texture_storage_3d", comment_free_shader_src));
    if (!output_texture_binding_) {
      return absl::InternalError("Bound output texture needed in shader.");
    } else {
      // Ensure if 3D texture output that we're using the appropriate output
      // type.
      const std::string output_format_str =
          GetOutputFormat("texture_storage_3d", comment_free_shader_src);
      if (output_format_str != "rg32uint") {
        return absl::InternalError(absl::StrFormat(
            "Output 3D texture format not supported. Should be rg32uint. Was: "
            "%s",
            output_format_str));
      }
    }
  } else {
    const std::string output_format_str =
        GetOutputFormat("texture_storage_2d", comment_free_shader_src);
    // Choose type of WebGPU output texture from our limited subset of supported
    // types.
    if (output_format_str == "rgba8unorm") {
      output_format_ = GpuBufferFormat::kRGBA32;
    } else if (output_format_str == "rgba32float") {
      output_format_ = GpuBufferFormat::kRGBAFloat128;
    } else if (output_format_str == "r32float") {
      output_format_ = GpuBufferFormat::kGrayFloat32;
    } else {
      return absl::InternalError(absl::StrFormat(
          "Output 2D texture format not supported. Should be rgba8unorm, "
          "rgba32float, or r32float. Was: %s",
          output_format_str));
    }
  }

  MP_ASSIGN_OR_RETURN(sampler_binding_,
                      GetBindingLocation("sampler", comment_free_shader_src));
  MP_ASSIGN_OR_RETURN(uniform_binding_,
                      GetBindingLocation("Params", comment_free_shader_src));
  MP_ASSIGN_OR_RETURN(
      input_texture_bindings_,
      GetBindingLocations("texture_2d", comment_free_shader_src));
  MP_ASSIGN_OR_RETURN(
      input_texture_3d_bindings_,
      GetBindingLocations("texture_3d", comment_free_shader_src));

  // Parse shader to grab Params uniform struct offsets
  MP_ASSIGN_OR_RETURN(param_offsets_, GetParamOffsets(comment_free_shader_src));
  params_data_ = std::make_unique<float[]>(param_offsets_.num_params);
  params_size_ = param_offsets_.num_params * sizeof(float);

  // Create the shader module.
  wgpu::ShaderSourceWGSL wgsl = {};
  wgsl.code = shader_source_.c_str();
  wgpu::ShaderModuleDescriptor shader_desc = {
      .nextInChain = &wgsl,
  };
  wgpu::ShaderModule module =
      service_->device().CreateShaderModule(&shader_desc);

  if (!module) {
    return absl::InternalError("Failed to create shader module.");
  }

  // Create the compute pipeline
  wgpu::ComputePipelineDescriptor pipeline_desc = {
      .compute =
          {
              .module = module,
              .entryPoint = "main",
              .constantCount = 0,
              .constants = nullptr,
          },
  };
  pipeline_future_ =
      WebGpuCreateComputePipelineAsync(service_->device(), &pipeline_desc);

  // Create a uniform buffer for the parameters
  wgpu::BufferDescriptor buffer_desc = {
      .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
      .size = params_size_,
  };
  params_ = service_->device().CreateBuffer(&buffer_desc);
  if (!params_) {
    return absl::InternalError("Failed to create params buffer.");
  }

  // And a default sampler in case we need that too
  wgpu::SamplerDescriptor sampler_desc = {
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
  };
  sampler_ = service_->device().CreateSampler(&sampler_desc);
  if (!sampler_) {
    return absl::InternalError("Failed to create sampler.");
  }
  return absl::OkStatus();
}

// Helper function for when we encounter an empty packet in one of our expected
// input sets.
void WebGpuShaderCalculator::HandleEmptyPacket(
    CalculatorContext* cc, int index, bool has_gpu_buffer_input,
    const std::string& stream_debug_name) {
  if (passthrough_first_buffer_on_empty_packets_) {
    ABSL_LOG_EVERY_N(WARNING, 100)
        << stream_debug_name << " input stream at id: " << index
        << " was empty. Passing through input buffer at index 0.";
    // We pass through first normal input first if we have any streams for that,
    // and only otherwise pass through first 3d input.
    if (has_gpu_buffer_input) {
      kOutput(cc).Send(*kInputBuffers(cc).begin());
    } else {
      kOutput3d(cc).Send(*kInputBuffers3d(cc).begin());
    }
  } else {
    ABSL_LOG_EVERY_N(WARNING, 100)
        << stream_debug_name << " input stream at id: " << index
        << " was empty. Skipping frame.";
  }
}

absl::Status WebGpuShaderCalculator::Process(CalculatorContext* cc) {
  ScopedWebGpuErrorHandler scoped_error_handler(
      service_, "WebGpuShaderCalculator::Process", cc->InputTimestamp());

  MP_ASSIGN_OR_RETURN(wgpu::ComputePipeline * pipeline, pipeline_future_.Get(),
                      _.SetCode(absl::StatusCode::kInternal).SetPrepend()
                          << "Failed to create pipeline: ");

  if (kInputWidth(cc).IsConnected() && !kInputWidth(cc).IsEmpty()) {
    output_width_ = kInputWidth(cc).Get();
  }
  if (kInputHeight(cc).IsConnected() && !kInputHeight(cc).IsEmpty()) {
    output_height_ = kInputHeight(cc).Get();
  }
  if (kInputDepth(cc).IsConnected() && !kInputDepth(cc).IsEmpty()) {
    output_depth_ = kInputDepth(cc).Get();
  }

  // Setup source textures from input gpu buffers
  const bool has_gpu_buffer_input = kInputBuffers(cc).Count() > 0;
  std::vector<WebGpuTextureView> src_textures;
  for (auto it = kInputBuffers(cc).begin(); it != kInputBuffers(cc).end();
       ++it) {
    auto packet_stream = *it;
    if (packet_stream.IsEmpty()) {
      if (it == kInputBuffers(cc).begin()) {
        ABSL_LOG_EVERY_N(WARNING, 100)
            << "GPU buffer input stream first packet was empty. "
            << "Skipping frame.";
      } else {
        HandleEmptyPacket(cc, std::distance(kInputBuffers(cc).begin(), it),
                          has_gpu_buffer_input, "GPU buffer");
      }
      return absl::OkStatus();
    }
    const auto& gpu_buffer = packet_stream.Get();
    src_textures.push_back(gpu_buffer.GetReadView<WebGpuTextureView>());
  }

  std::vector<WebGpuTextureView> src_textures_3d;
  for (auto it = kInputBuffers3d(cc).begin(); it != kInputBuffers3d(cc).end();
       ++it) {
    auto packet_stream = *it;
    if (packet_stream.IsEmpty()) {
      if (it == kInputBuffers3d(cc).begin() && !has_gpu_buffer_input) {
        ABSL_LOG_EVERY_N(WARNING, 100)
            << "3D texture buffer input stream first packet was empty,"
            << "and no GPU buffer input stream attached. Skipping "
            << "frame.";
      } else {
        HandleEmptyPacket(cc, std::distance(kInputBuffers3d(cc).begin(), it),
                          has_gpu_buffer_input, "3D texture buffer");
      }
      return absl::OkStatus();
    }
    const auto& texture_3d = packet_stream.Get();
    src_textures_3d.push_back(texture_3d.GetReadView());
  }

  std::vector<float> src_floats;
  for (auto it = kInputFloats(cc).begin(); it != kInputFloats(cc).end(); ++it) {
    auto packet_stream = *it;
    if (packet_stream.IsEmpty()) {
      HandleEmptyPacket(cc, std::distance(kInputFloats(cc).begin(), it),
                        has_gpu_buffer_input, "Float uniform");
      return absl::OkStatus();
    }
    src_floats.push_back(packet_stream.Get());
  }

  std::vector<std::vector<float>> src_float_vecs;
  for (auto it = kInputFloatVecs(cc).begin(); it != kInputFloatVecs(cc).end();
       ++it) {
    auto packet_stream = *it;
    if (packet_stream.IsEmpty()) {
      HandleEmptyPacket(cc, std::distance(kInputFloatVecs(cc).begin(), it),
                        has_gpu_buffer_input, "Float vector uniform");
      return absl::OkStatus();
    }
    src_float_vecs.push_back(packet_stream.Get());
  }

  // Destination size default is 640x480 if no inputs, and otherwise, we'll
  // use size of first input.
  int width = 640;
  int height = 480;
  int depth = 0;

  if (!src_textures.empty()) {
    width = src_textures[0].width();
    height = src_textures[0].height();
  } else if (!src_textures_3d.empty()) {
    width = src_textures_3d[0].width();
    height = src_textures_3d[0].height();
    depth = src_textures_3d[0].depth();
  }

  if (output_width_) width = output_width_.value();
  if (output_height_) height = output_height_.value();
  if (output_depth_) depth = output_depth_.value();

  if (kOutput(cc).IsConnected() && depth > 0) {
    depth = 0;
    ABSL_LOG(WARNING)
        << "Forcing depth to 0 because output tag indicates that we "
        << "are rendering to a 2D texture, not a 3D texture.";
  }

  MP_RETURN_IF_ERROR(WebGpuBindAndRender(cc, *pipeline, width, height, depth,
                                         src_textures, src_textures_3d,
                                         src_floats, src_float_vecs));
  return absl::OkStatus();
}

absl::Status WebGpuShaderCalculator::WebGpuBindAndRender(
    CalculatorContext* cc, const wgpu::ComputePipeline& pipeline, int width,
    int height, int depth, const std::vector<WebGpuTextureView>& src_textures,
    const std::vector<WebGpuTextureView>& src_textures_3d,
    const std::vector<float>& src_floats,
    const std::vector<std::vector<float>>& src_float_vecs) {
  ScopedWebGpuErrorHandler scoped_error_handler(
      service_, "WebGpuShaderCalculator::WebGpuBindAndRender",
      cc->InputTimestamp());
  // Setup rendering to new destination GpuBuffer or WebGpuTextureBuffer3d, if
  // not rendering to screen. TODO: Allow for different output formats.
  if (depth == 0) {
    // Standard 2d texture rendering
    GpuBuffer out_buffer(width, height, output_format_);
    WebGpuTextureView out_view = out_buffer.GetWriteView<WebGpuTextureView>();
    MP_RETURN_IF_ERROR(WebGpuBindAndRenderToView(
        cc, pipeline, width, height, depth, src_textures, src_textures_3d,
        src_floats, src_float_vecs, out_view));
    kOutput(cc).Send(std::move(out_buffer));
  } else {
    // Special 3d texture rendering
    auto out_buffer = WebGpuTextureBuffer3d::Create(
        width, height, depth, WebGpuTextureFormat3d::kRG32Uint);
    WebGpuTextureView out_view = out_buffer->GetWriteView();
    MP_RETURN_IF_ERROR(WebGpuBindAndRenderToView(
        cc, pipeline, width, height, depth, src_textures, src_textures_3d,
        src_floats, src_float_vecs, out_view));
    kOutput3d(cc).Send(std::move(out_buffer));
  }
  return absl::OkStatus();
}

absl::Status WebGpuShaderCalculator::WebGpuBindAndRenderToView(
    CalculatorContext* cc, const wgpu::ComputePipeline& pipeline, int width,
    int height, int depth, const std::vector<WebGpuTextureView>& src_textures,
    const std::vector<WebGpuTextureView>& src_textures_3d,
    const std::vector<float>& src_floats,
    const std::vector<std::vector<float>>& src_float_vecs,
    WebGpuTextureView& out_view) {
  ScopedWebGpuErrorHandler scoped_error_handler(
      service_, "WebGpuShaderCalculator::WebGpuBindAndRenderToView",
      cc->InputTimestamp());
  const wgpu::Device& device = service_->device();

  // Update Params struct
  if (param_offsets_.output_size_offset) {
    reinterpret_cast<int32_t*>(
        params_data_.get())[param_offsets_.output_size_offset.value()] = width;
    reinterpret_cast<int32_t*>(
        params_data_.get())[param_offsets_.output_size_offset.value() + 1] =
        height;
    reinterpret_cast<int32_t*>(
        params_data_.get())[param_offsets_.output_size_offset.value() + 2] =
        depth;
  }
  if (param_offsets_.time_offset) {
    params_data_[param_offsets_.time_offset.value()] =
        cc->InputTimestamp().Seconds();
  }

  RET_CHECK_LE(src_floats.size(), param_offsets_.float_offsets.size())
      << "Must have at least as many float uniforms as float inputs. "
         "Potentially there is a mismatch between the shader and the graph "
         "config.";

  for (int i = 0; i < src_floats.size(); i++) {
    params_data_[param_offsets_.float_offsets[i]] = src_floats[i];
  }

  RET_CHECK_LE(src_float_vecs.size(), param_offsets_.float_vec_offsets.size())
      << "Must have at least as many float vector uniforms as float vector "
         "inputs. Potentially there is a mismatch between the shader and the "
         "graph config.";

  for (int i = 0; i < src_float_vecs.size(); i++) {
    int offset = param_offsets_.float_vec_offsets[i];
    for (const float val : src_float_vecs[i]) {
      params_data_[offset] = val;
      offset++;
    }
  }

  device.GetQueue().WriteBuffer(params_, 0, params_data_.get(), params_size_);

  // Create the bind group; here's where we bind everything.
  // TODO: Optimize, and allow for many more binding types, including
  // uniforms.
  // For input textures
  uint32_t num_entries = src_textures.size() + src_textures_3d.size();
  if (sampler_binding_) num_entries++;  // For sampler
  num_entries++;                        // For destination texture
  if (uniform_binding_) num_entries++;  // For Params

  std::vector<wgpu::BindGroupEntry> entries;
  entries.reserve(num_entries);

  if (sampler_binding_) {
    entries.push_back({
        .binding = static_cast<uint32_t>(sampler_binding_.value()),
        .sampler = sampler_,
    });
  }
  int in_texture_count = 0;
  for (auto& src_texture : src_textures) {
    entries.push_back({
        .binding =
            static_cast<uint32_t>(input_texture_bindings_[in_texture_count]),
        .textureView = src_texture.texture().CreateView(),
    });
    in_texture_count++;
  }
  in_texture_count = 0;
  for (auto& src_texture_3d : src_textures_3d) {
    entries.push_back({
        .binding =
            static_cast<uint32_t>(input_texture_3d_bindings_[in_texture_count]),
        .textureView = src_texture_3d.texture().CreateView(),
    });
    in_texture_count++;
  }
  entries.push_back({
      .binding = static_cast<uint32_t>(output_texture_binding_.value()),
      .textureView = out_view.texture().CreateView(),
  });
  if (uniform_binding_) {
    entries.push_back({
        .binding = static_cast<uint32_t>(uniform_binding_.value()),
        .buffer = params_,
        .size = params_size_,
    });
  }

  wgpu::BindGroupDescriptor bind_group_desc = {
      .layout = pipeline.GetBindGroupLayout(0),
      .entryCount = num_entries,
      .entries = entries.data(),
  };
  wgpu::BindGroup bind_group = device.CreateBindGroup(&bind_group_desc);
  if (!bind_group) {
    return absl::InternalError("Failed to create bind group.");
  }

  // Round up the number of workgroups to cover the whole texture.
  const uint32_t num_groups_x =
      (out_view.width() + workgroup_size_x_ - 1) / workgroup_size_x_;
  const uint32_t num_groups_y =
      (out_view.height() + workgroup_size_y_ - 1) / workgroup_size_y_;

  // For views onto 2d textures, depth() will still be 1, by default.
  const uint32_t num_groups_z =
      (out_view.depth() + workgroup_size_z_ - 1) / workgroup_size_z_;

  // Create and submit a command buffer that dispatches the compute shader.
  wgpu::ComputePassDescriptor pass_descriptor;
  if (profile_) {
    pass_descriptor = {
        .timestampWrites = &timestamp_writes_,
    };
  } else {
    repetitions_ = 1;           // No repetitions if not profiling
    skip_starting_frames_ = 0;  // No frame skipping if not profiling
  }
  auto command_encoder = device.CreateCommandEncoder();
  for (int i = 0; i < repetitions_ + skip_starting_frames_; ++i) {
    auto pass_encoder = (profile_ && i == skip_starting_frames_)
                            ? command_encoder.BeginComputePass(&pass_descriptor)
                            : command_encoder.BeginComputePass();
    pass_encoder.SetPipeline(pipeline);
    pass_encoder.SetBindGroup(0, bind_group);
    pass_encoder.DispatchWorkgroups(num_groups_x, num_groups_y, num_groups_z);
    pass_encoder.End();
  }
  if (profile_) {
    command_encoder.ResolveQuerySet(query_set_, /*firstQuery=*/0, 2,
                                    query_buffer_, /*destinationOffset=*/0);
    command_encoder.CopyBufferToBuffer(query_buffer_, /*offset=*/0, dst_buffer_,
                                       /*offset=*/0, kQueryBufferByteSize);
  }
  wgpu::CommandBuffer command_buffers[] = {
      command_encoder.Finish(),
  };
  device.GetQueue().Submit(std::size(command_buffers), command_buffers);

  if (profile_) {
    ExposeProfilingResults(cc->NodeName().c_str(), dst_buffer_.Get(),
                           repetitions_);
  }
  return absl::OkStatus();
}

absl::Status WebGpuShaderCalculator::Close(CalculatorContext* cc) {
  service_ = nullptr;
  pipeline_future_.Reset();
  return absl::OkStatus();
}

}  // namespace mediapipe

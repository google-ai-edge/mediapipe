#include <algorithm>
#include <memory>
#include <vector>

#include "mediapipe/framework/formats/tensor/backend.h"
#include "mediapipe/framework/formats/tensor/tensor2.h"
#include "mediapipe/framework/formats/tensor/views/buffer.h"
#include "mediapipe/framework/formats/tensor/views/cpu_buffer.h"
#include "third_party/FP16/include/fp16.h"

namespace mediapipe {
namespace {

template <class SourceType, class DestinationType>
auto ConverterCheckFunction() {
  return
      [](const Tensor2& tensor, uint64_t source_descriptor_type_id,
         const Tensor2::ViewDescriptor& source_base_descriptor,
         uint64_t destination_descriptor_type_id,
         const Tensor2::ViewDescriptor& destination_base_descriptor) -> bool {
        if (source_descriptor_type_id != TensorCpuView::kId ||
            destination_descriptor_type_id != TensorCpuView::kId)
          return false;
        auto source_descriptor =
            static_cast<const TensorCpuViewDescriptor&>(source_base_descriptor);
        auto destination_descriptor =
            static_cast<const TensorCpuViewDescriptor&>(
                destination_base_descriptor);
        return source_descriptor.buffer.format ==
                   TensorTypeToFormat<SourceType>::value &&
               destination_descriptor.buffer.format ==
                   TensorTypeToFormat<DestinationType>::value;
      };
}

template <class SourceType, class DestinationType>
auto ConvertFunction() {
  return [](const Tensor2& tensor, const Tensor2::View& source_base_view,
            const Tensor2::View& destination_base_view) -> bool {
    auto source = source_base_view.DownCast<TensorCpuView>();
    auto destination = destination_base_view.DownCast<TensorCpuView>();
    if (source->descriptor().buffer.format ==
        destination->descriptor().buffer.format) {
      std::memcpy(
          destination->data<void>(), source->data<void>(),
          TensorBufferSize(destination->descriptor().buffer, tensor.shape()));
    } else {
      auto source_pointer = source->data<SourceType>();
      auto destination_pointer = destination->data<DestinationType>();
      for (int i = 0; i < tensor.shape().NumElements(); i++) {
        *destination_pointer++ =
            GpuLikeTypeCast<DestinationType>(*source_pointer++);
      }
    }
    return true;
  };
}

#define REGISTER_CONVERTER(SourceType, DestinationType)       \
  TENSOR_REGISTER_CONVERTER(                                  \
      {ConverterCheckFunction<SourceType, DestinationType>(), \
       ConvertFunction<SourceType, DestinationType>()});

REGISTER_CONVERTER(float, Float16);
REGISTER_CONVERTER(float, int8_t);
REGISTER_CONVERTER(float, uint8_t);
REGISTER_CONVERTER(float, int16_t);
REGISTER_CONVERTER(float, uint16_t);
REGISTER_CONVERTER(float, int32_t);
REGISTER_CONVERTER(float, uint32_t);

REGISTER_CONVERTER(Float16, float);
REGISTER_CONVERTER(Float16, int8_t);
REGISTER_CONVERTER(Float16, uint8_t);
REGISTER_CONVERTER(Float16, int16_t);
REGISTER_CONVERTER(Float16, uint16_t);
REGISTER_CONVERTER(Float16, int32_t);
REGISTER_CONVERTER(Float16, uint32_t);

REGISTER_CONVERTER(int8_t, float);
REGISTER_CONVERTER(int8_t, Float16);
REGISTER_CONVERTER(int8_t, uint8_t);
REGISTER_CONVERTER(int8_t, int16_t);
REGISTER_CONVERTER(int8_t, uint16_t);
REGISTER_CONVERTER(int8_t, int32_t);
REGISTER_CONVERTER(int8_t, uint32_t);

REGISTER_CONVERTER(uint8_t, float);
REGISTER_CONVERTER(uint8_t, Float16);
REGISTER_CONVERTER(uint8_t, int8_t);
REGISTER_CONVERTER(uint8_t, int16_t);
REGISTER_CONVERTER(uint8_t, uint16_t);
REGISTER_CONVERTER(uint8_t, int32_t);
REGISTER_CONVERTER(uint8_t, uint32_t);

REGISTER_CONVERTER(int16_t, float);
REGISTER_CONVERTER(int16_t, Float16);
REGISTER_CONVERTER(int16_t, int8_t);
REGISTER_CONVERTER(int16_t, uint8_t);
REGISTER_CONVERTER(int16_t, uint16_t);
REGISTER_CONVERTER(int16_t, uint32_t);
REGISTER_CONVERTER(int16_t, uint32_t);

REGISTER_CONVERTER(uint16_t, float);
REGISTER_CONVERTER(uint16_t, Float16);
REGISTER_CONVERTER(uint16_t, int8_t);
REGISTER_CONVERTER(uint16_t, uint8_t);
REGISTER_CONVERTER(uint16_t, int16_t);
REGISTER_CONVERTER(uint16_t, int32_t);
REGISTER_CONVERTER(uint16_t, uint32_t);

REGISTER_CONVERTER(int32_t, float);
REGISTER_CONVERTER(int32_t, Float16);
REGISTER_CONVERTER(int32_t, int8_t);
REGISTER_CONVERTER(int32_t, uint8_t);
REGISTER_CONVERTER(int32_t, int16_t);
REGISTER_CONVERTER(int32_t, uint16_t);
REGISTER_CONVERTER(int32_t, uint32_t);

REGISTER_CONVERTER(uint32_t, float);
REGISTER_CONVERTER(uint32_t, Float16);
REGISTER_CONVERTER(uint32_t, int8_t);
REGISTER_CONVERTER(uint32_t, uint8_t);
REGISTER_CONVERTER(uint32_t, int16_t);
REGISTER_CONVERTER(uint32_t, uint16_t);
REGISTER_CONVERTER(uint32_t, int32_t);

template <class DestinationType>
auto DequantizationCheckFunction() {
  return
      [](const Tensor2& tensor, uint64_t source_descriptor_type_id,
         const Tensor2::ViewDescriptor& source_base_descriptor,
         uint64_t destination_descriptor_type_id,
         const Tensor2::ViewDescriptor& destination_base_descriptor) -> bool {
        if (source_descriptor_type_id != TensorCpuView::kId ||
            destination_descriptor_type_id != TensorCpuView::kId)
          return false;
        auto source_descriptor =
            static_cast<const TensorCpuViewDescriptor&>(source_base_descriptor);
        auto destination_descriptor =
            static_cast<const TensorCpuViewDescriptor&>(
                destination_base_descriptor);
        return source_descriptor.buffer.format ==
                   TensorBufferDescriptor::Format::kQuantizedInt8 &&
               destination_descriptor.buffer.format ==
                   TensorTypeToFormat<DestinationType>::value;
      };
}

template <class DestinationType>
auto DequantizationConvertFunction() {
  return [](const Tensor2& tensor, const Tensor2::View& source_base_view,
            const Tensor2::View& destination_base_view) -> bool {
    auto source = source_base_view.DownCast<TensorCpuView>();
    auto destination = destination_base_view.DownCast<TensorCpuView>();
    auto source_pointer = source->data<int8_t>();
    auto destination_pointer = destination->data<DestinationType>();
    int zero_point =
        source->descriptor().buffer.quantization_parameters.zero_point;
    float scale = source->descriptor().buffer.quantization_parameters.scale;
    for (int i = 0; i < tensor.shape().NumElements(); i++) {
      *destination_pointer++ = static_cast<DestinationType>(
          (*source_pointer++ - zero_point) * scale);
    }
    return true;
  };
}

#define REGISTER_DEQUANTIZATION_CONVERTER(DestinationType) \
  TENSOR_REGISTER_CONVERTER(                               \
      {DequantizationCheckFunction<DestinationType>(),     \
       DequantizationConvertFunction<DestinationType>()});

REGISTER_DEQUANTIZATION_CONVERTER(float);
REGISTER_DEQUANTIZATION_CONVERTER(Float16);
REGISTER_DEQUANTIZATION_CONVERTER(int8_t);
REGISTER_DEQUANTIZATION_CONVERTER(uint8_t);
REGISTER_DEQUANTIZATION_CONVERTER(int16_t);
REGISTER_DEQUANTIZATION_CONVERTER(uint16_t);
REGISTER_DEQUANTIZATION_CONVERTER(int32_t);
REGISTER_DEQUANTIZATION_CONVERTER(uint32_t);

template <class SourceType>
auto QuantizationCheckFunction() {
  return
      [](const Tensor2& tensor, uint64_t source_descriptor_type_id,
         const Tensor2::ViewDescriptor& source_base_descriptor,
         uint64_t destination_descriptor_type_id,
         const Tensor2::ViewDescriptor& destination_base_descriptor) -> bool {
        if (source_descriptor_type_id != TensorCpuView::kId ||
            destination_descriptor_type_id != TensorCpuView::kId)
          return false;
        auto source_descriptor =
            static_cast<const TensorCpuViewDescriptor&>(source_base_descriptor);
        auto destination_descriptor =
            static_cast<const TensorCpuViewDescriptor&>(
                destination_base_descriptor);
        bool same = source_descriptor.buffer.format ==
                        TensorTypeToFormat<SourceType>::value &&
                    destination_descriptor.buffer.format ==
                        TensorBufferDescriptor::Format::kQuantizedInt8;
        return same;
      };
}

template <class SourceType>
auto QuantizationConvertFunction() {
  return [](const Tensor2& tensor, const Tensor2::View& source_base_view,
            const Tensor2::View& destination_base_view) -> bool {
    auto source = source_base_view.DownCast<TensorCpuView>();
    auto destination = destination_base_view.DownCast<TensorCpuView>();
    auto source_pointer = source->data<SourceType>();
    auto destination_pointer = destination->data<int8_t>();
    int zero_point =
        destination->descriptor().buffer.quantization_parameters.zero_point;
    float scale =
        destination->descriptor().buffer.quantization_parameters.scale;
    for (int i = 0; i < tensor.shape().NumElements(); i++) {
      *destination_pointer++ =
          static_cast<int8_t>(*source_pointer++ / scale + zero_point);
    }
    return true;
  };
}

#define REGISTER_QUANTIZATION_CONVERTER(SourceType)                   \
  TENSOR_REGISTER_CONVERTER({QuantizationCheckFunction<SourceType>(), \
                             QuantizationConvertFunction<SourceType>()});

REGISTER_QUANTIZATION_CONVERTER(float);
REGISTER_QUANTIZATION_CONVERTER(Float16);
REGISTER_QUANTIZATION_CONVERTER(int8_t);
REGISTER_QUANTIZATION_CONVERTER(uint8_t);
REGISTER_QUANTIZATION_CONVERTER(int16_t);
REGISTER_QUANTIZATION_CONVERTER(uint16_t);
REGISTER_QUANTIZATION_CONVERTER(int32_t);
REGISTER_QUANTIZATION_CONVERTER(uint32_t);

}  // namespace
}  // namespace mediapipe

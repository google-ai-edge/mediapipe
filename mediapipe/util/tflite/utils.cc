#include "mediapipe/util/tflite/utils.h"

#include "tensorflow/lite/c/common.h"

namespace mediapipe::util::tflite {

bool IsDynamicTensor(const TfLiteTensor& tensor) {
  for (int i = 0; i < tensor.dims->size; ++i) {
    if (tensor.dims->data[i] == -1) {
      return true;
    }
  }
  return false;
}
}  // namespace mediapipe::util::tflite

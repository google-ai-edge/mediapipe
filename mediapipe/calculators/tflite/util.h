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

#ifndef MEDIAPIPE_CALCULATORS_TFLITE_UTIL_H_
#define MEDIAPIPE_CALCULATORS_TFLITE_UTIL_H_

#include "tensorflow/lite/interpreter.h"

#define RET_CHECK_CALL(call)                                     \
  do {                                                           \
    const auto status = (call);                                  \
    if (ABSL_PREDICT_FALSE(!status.ok()))                        \
      return ::mediapipe::InternalError(status.message()); \
  } while (0);

namespace mediapipe {
    class TfLiteTensorContainer {
    private:
        TfLiteTensor tensor_;
        std::unique_ptr<TfLiteIntArray> dims_;
        std::unique_ptr<char[]> data_;

        //Free internal memory
        void FreeTensor() {
            tensor_.dims = 0;
            tensor_.data.raw = 0;
            dims_.reset();
            data_.reset();
        }
        //Copy data from source tensor
        void CopyTensor(const TfLiteTensor& tensor) {
            //Free internal memory for copy new data
            FreeTensor();

            //Copy data from source to internal member
            dims_.reset(TfLiteIntArrayCreate((tensor.dims)->size));
            memcpy(dims_->data, (tensor.dims)->data, sizeof(int)*((tensor.dims)->size));
            data_ = absl::make_unique<char[]>(tensor.bytes);
            memcpy(data_.get(), tensor.data.raw, tensor.bytes);
            memcpy(&tensor_, &tensor, sizeof(TfLiteTensor));
            tensor_.dims = dims_.get();
            tensor_.data.raw = data_.get();
        }
    public:
        TfLiteTensorContainer(const TfLiteTensor& tensor) {
            CopyTensor(tensor);
        }
        //Copy constructor
        TfLiteTensorContainer(const TfLiteTensorContainer& tensor_ctn) {
            CopyTensor(tensor_ctn.getTensor());
        }

        //Destructor
        ~TfLiteTensorContainer() {
            FreeTensor();
        }

        // Get tensor
        const TfLiteTensor& getTensor() const {
            return tensor_;
        }

        //Assign operator
        TfLiteTensorContainer & operator= ( TfLiteTensorContainer tensor_ctn){
            CopyTensor(tensor_ctn.getTensor());

            return *this;
        }
        TfLiteTensorContainer & operator= ( const TfLiteTensorContainer & tensor_ctn) {
            CopyTensor(tensor_ctn.getTensor());

            return *this;
        }
    };
}

#endif  // MEDIAPIPE_CALCULATORS_TFLITE_UTIL_H_

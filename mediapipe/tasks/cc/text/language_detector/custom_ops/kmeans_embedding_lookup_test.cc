#include "mediapipe/tasks/cc/text/language_detector/custom_ops/kmeans_embedding_lookup.h"

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>

#include "absl/log/absl_check.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"

namespace mediapipe::tflite_operations {
namespace {

using ::testing::ElementsAreArray;
using ::tflite::ArrayFloatNear;

// Helper class for testing the op.
class KmeansEmbeddingLookupModel : public tflite::SingleOpModel {
 public:
  explicit KmeansEmbeddingLookupModel(
      std::initializer_list<int> input_shape,
      std::initializer_list<int> encoding_table_shape,
      std::initializer_list<int> codebook_shape,
      std::initializer_list<int> output_shape) {
    // Setup the model inputs and the interpreter.
    output_ = AddOutput({tflite::TensorType_FLOAT32, output_shape});
    SetCustomOp("KmeansEmbeddingLookup", std::vector<uint8_t>(),
                Register_KmeansEmbeddingLookup);
    BuildInterpreter({input_shape, encoding_table_shape, codebook_shape});
  }

  TfLiteStatus SetUpInputTensor(const std::vector<int>& input,
                                const std::vector<uint8_t>& encoding_table,
                                const std::vector<float>& codebook) {
    PopulateTensor<int>(input_, {input});
    PopulateTensor<uint8_t>(encoding_table_, {encoding_table});
    PopulateTensor<float>(codebook_, {codebook});
    return interpreter_->AllocateTensors();
  }

  void Invoke(const std::vector<int>& input,
              const std::vector<uint8_t>& encoding_table,
              const std::vector<float>& codebook) {
    ABSL_CHECK_EQ(SetUpInputTensor(input, encoding_table, codebook), kTfLiteOk);
    ABSL_CHECK_EQ(SingleOpModel::Invoke(), kTfLiteOk);
  }

  TfLiteStatus InvokeUnchecked(const std::vector<int>& input,
                               const std::vector<uint8_t>& encoding_table,
                               const std::vector<float>& codebook) {
    TfLiteStatus allocation_status =
        SetUpInputTensor(input, encoding_table, codebook);
    if (allocation_status != kTfLiteOk) {
      return allocation_status;
    }
    return SingleOpModel::Invoke();
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_ = AddInput(tflite::TensorType_INT32);
  int encoding_table_ = AddInput(tflite::TensorType_UINT8);
  int codebook_ = AddInput(tflite::TensorType_FLOAT32);
  int output_;
};

template <typename T>
std::vector<T> FlattenVector2D(std::vector<std::vector<T>> input_vec) {
  std::vector<T> output_vec(input_vec.size() * input_vec[0].size());
  for (int i = 0, k = 0; i < input_vec.size(); i++) {
    for (int j = 0; j < input_vec[i].size(); j++, k++) {
      output_vec[k] = input_vec[i][j];
    }
  }
  return output_vec;
}

class KmeansEmbeddingLookupTestWithSampleInputs : public ::testing::Test {
 public:
  KmeansEmbeddingLookupTestWithSampleInputs() {
    input_ = std::vector<int>({1, 2, 3, 0, 0});
    encoding_table_ = std::vector<std::vector<uint8_t>>(
        {{0, 0}, {1, 1}, {1, 2}, {1, 0}, {1, 0}, {2, 0}});
    codebook_ =
        std::vector<std::vector<float>>({{0.0, 0.0}, {7.0, 7.0}, {7.0, 0.0}});
    expected_output_ = std::vector<float>(
        // The output is the average of the embeddings at the three indices
        // (1, 2, 3).
        {7.0, 7.0, 4.66667, 2.33333});
  }

 protected:
  std::vector<int> input_;
  std::vector<std::vector<uint8_t>> encoding_table_;
  std::vector<std::vector<float>> codebook_;
  std::vector<float> expected_output_;
};

TEST_F(KmeansEmbeddingLookupTestWithSampleInputs, ReturnsCorrectly) {
  // Check if the expected output is returned
  KmeansEmbeddingLookupModel m(/*input_shape=*/{1, 5},
                               /*encoding_table_shape=*/{6, 2},
                               /*codebook_shape=*/{3, 2},
                               /*output_shape=*/{1, 4});

  m.Invoke(input_, FlattenVector2D<uint8_t>(encoding_table_),
           FlattenVector2D<float>(codebook_));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(expected_output_, 1e-5)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4}));
}

TEST_F(KmeansEmbeddingLookupTestWithSampleInputs,
       HandlesNegativeValuesInCodebook) {
  KmeansEmbeddingLookupModel m(/*input_shape=*/{1, 4},
                               /*encoding_table_shape=*/{4, 2},
                               /*codebook_shape=*/{4, 3},
                               /*output_shape=*/{1, 6});
  std::vector<int> input = std::vector<int>({2, 2, 1, 3});
  std::vector<std::vector<uint8_t>> encoding_table =
      std::vector<std::vector<uint8_t>>({{0, 0}, {1, 2}, {3, 0}, {2, 3}});
  std::vector<std::vector<float>> codebook = std::vector<std::vector<float>>(
      {{5.0, 2.0, 3.0}, {8.0, 2.0, 4.0}, {1.2, 2.4, 3.6}, {0.5, -2.0, 1.0}});
  m.Invoke(input, FlattenVector2D<uint8_t>(encoding_table),
           FlattenVector2D<float>(codebook));
  std::vector<float> expected_output =
      std::vector<float>({2.55, 0.1, 2.4, 2.925, 1.1, 2.65});
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(expected_output, 1e-5)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 6}));
}

TEST_F(KmeansEmbeddingLookupTestWithSampleInputs, IgnoresIndicesAfterZero) {
  KmeansEmbeddingLookupModel m(/*input_shape=*/{1, 4},
                               /*encoding_table_shape=*/{4, 2},
                               /*codebook_shape=*/{4, 3},
                               /*output_shape=*/{1, 6});
  std::vector<int> input = std::vector<int>({2, 2, 0, 3});
  std::vector<std::vector<uint8_t>> encoding_table =
      std::vector<std::vector<uint8_t>>({{0, 0}, {1, 2}, {3, 0}, {2, 3}});
  std::vector<std::vector<float>> codebook = std::vector<std::vector<float>>(
      {{5.0, 2.0, 3.0}, {8.0, 2.0, 4.0}, {1.2, 2.4, 3.6}, {0.5, -2.0, 1.0}});
  m.Invoke(input, FlattenVector2D<uint8_t>(encoding_table),
           FlattenVector2D<float>(codebook));
  std::vector<float> expected_output =
      std::vector<float>({0.5, -2.0, 1.0, 5.0, 2.0, 3.0});
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear(expected_output, 1e-5)));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 6}));
}

TEST(KmeansEmbeddingLookupTest, ThrowsErrorWhenGivenInvalidInputBatchSize) {
  // Check that the op errors out when the batch size is greater than 1.
  KmeansEmbeddingLookupModel m(/*input_shape=*/{2, 1},
                               /*encoding_table_shape=*/{1, 1},
                               /*codebook shape=*/{1, 2},
                               /*output_shape=*/{2, 2});

  EXPECT_EQ(m.InvokeUnchecked(/*input=*/{1, 1},
                              /*encoding_table=*/{0},
                              /*codebook=*/{2.3, 4.5}),
            kTfLiteError);
}

}  // namespace
}  // namespace mediapipe::tflite_operations

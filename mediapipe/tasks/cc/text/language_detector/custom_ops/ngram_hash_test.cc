/* Copyright 2023 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/cc/text/language_detector/custom_ops/ngram_hash.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/types/optional.h"
#include "flatbuffers/flexbuffers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/cc/text/language_detector/custom_ops/utils/hash/murmur.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_util.h"

namespace mediapipe::tflite_operations {
namespace {

using ::flexbuffers::Builder;
using ::mediapipe::tasks::text::language_detector::custom_ops::hash::
    MurmurHash64WithSeed;
using ::testing::ElementsAreArray;
using ::testing::Message;

// Helper class for testing the op.
class NGramHashModel : public tflite::SingleOpModel {
 public:
  explicit NGramHashModel(const uint64_t seed,
                          const std::vector<int>& ngram_lengths,
                          const std::vector<int>& vocab_sizes,
                          const absl::optional<int> max_splits = std::nullopt) {
    // Setup the model inputs.
    Builder fbb;
    size_t start = fbb.StartMap();
    fbb.UInt("seed", seed);
    {
      size_t start = fbb.StartVector("ngram_lengths");
      for (const int& ngram_len : ngram_lengths) {
        fbb.Int(ngram_len);
      }
      fbb.EndVector(start, /*typed=*/true, /*fixed=*/false);
    }
    {
      size_t start = fbb.StartVector("vocab_sizes");
      for (const int& vocab_size : vocab_sizes) {
        fbb.Int(vocab_size);
      }
      fbb.EndVector(start, /*typed=*/true, /*fixed=*/false);
    }
    if (max_splits) {
      fbb.Int("max_splits", *max_splits);
    }
    fbb.EndMap(start);
    fbb.Finish();
    output_ = AddOutput({tflite::TensorType_INT32, {}});
    SetCustomOp("NGramHash", fbb.GetBuffer(), Register_NGRAM_HASH);
    BuildInterpreter({GetShape(input_)});
  }

  void SetupInputTensor(const std::string& input) {
    PopulateStringTensor(input_, {input});
    ABSL_CHECK(interpreter_->AllocateTensors() == kTfLiteOk)
        << "Cannot allocate tensors";
  }

  void Invoke(const std::string& input) {
    SetupInputTensor(input);
    ABSL_CHECK_EQ(SingleOpModel::Invoke(), kTfLiteOk);
  }

  TfLiteStatus InvokeUnchecked(const std::string& input) {
    SetupInputTensor(input);
    return SingleOpModel::Invoke();
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_ = AddInput(tflite::TensorType_STRING);
  int output_;
};

TEST(NGramHashTest, ReturnsExpectedValueWhenInputIsSane) {
  // Checks that the op returns the expected value when the input is sane.
  // Also checks that when `max_splits` is not specified, the entire string is
  // tokenized.
  const uint64_t kSeed = 123;
  const std::vector<int> vocab_sizes({100, 200});
  std::vector<int> ngram_lengths({1, 2});
  const std::vector<std::string> testcase_inputs({
      "hi",
      "wow",
      "!",
      "HI",
  });

  // A hash function that maps the given string to an index in the embedding
  // table denoted by `vocab_idx`.
  auto hash = [vocab_sizes, kSeed](std::string str, const int vocab_idx) {
    const auto hash_value =
        MurmurHash64WithSeed(str.c_str(), str.size(), kSeed);
    return static_cast<int>((hash_value % vocab_sizes[vocab_idx]) + 1);
  };
  const std::vector<std::vector<int>> expected_testcase_outputs(
      {{
           // Unigram & Bigram output for "hi".
           hash("^", 0),
           hash("h", 0),
           hash("i", 0),
           hash("$", 0),
           hash("^h", 1),
           hash("hi", 1),
           hash("i$", 1),
           hash("$", 1),
       },
       {
           // Unigram & Bigram output for "wow".
           hash("^", 0),
           hash("w", 0),
           hash("o", 0),
           hash("w", 0),
           hash("$", 0),
           hash("^w", 1),
           hash("wo", 1),
           hash("ow", 1),
           hash("w$", 1),
           hash("$", 1),
       },
       {
           // Unigram & Bigram output for "!" (which will get replaced by " ").
           hash("^", 0),
           hash(" ", 0),
           hash("$", 0),
           hash("^ ", 1),
           hash(" $", 1),
           hash("$", 1),
       },
       {
           // Unigram & Bigram output for "HI" (which will get lower-cased).
           hash("^", 0),
           hash("h", 0),
           hash("i", 0),
           hash("$", 0),
           hash("^h", 1),
           hash("hi", 1),
           hash("i$", 1),
           hash("$", 1),
       }});

  NGramHashModel m(kSeed, ngram_lengths, vocab_sizes);
  for (int test_idx = 0; test_idx < testcase_inputs.size(); test_idx++) {
    const std::string& testcase_input = testcase_inputs[test_idx];
    m.Invoke(testcase_input);
    SCOPED_TRACE(Message() << "Where the testcases' input is: "
                           << testcase_input);
    EXPECT_THAT(m.GetOutput<int>(),
                ElementsAreArray(expected_testcase_outputs[test_idx]));
    EXPECT_THAT(m.GetOutputShape(),
                ElementsAreArray(
                    {/*batch_size=*/1, static_cast<int>(ngram_lengths.size()),
                     static_cast<int>(testcase_input.size()) + /*padding*/ 2}));
  }
}

TEST(NGramHashTest, ReturnsExpectedValueWhenMaxSplitsIsSpecified) {
  // Checks that the op returns the expected value when the input is correct
  // when `max_splits` is specified.
  const uint64_t kSeed = 123;
  const std::vector<int> vocab_sizes({100, 200});
  std::vector<int> ngram_lengths({1, 2});

  const std::string testcase_input = "wow";
  const std::vector<int> max_splits({2, 3, 4, 5, 6});

  // A hash function that maps the given string to an index in the embedding
  // table denoted by `vocab_idx`.
  auto hash = [=](std::string str, const int vocab_idx) {
    const auto hash_value =
        MurmurHash64WithSeed(str.c_str(), str.size(), kSeed);
    return static_cast<int>((hash_value % vocab_sizes[vocab_idx]) + 1);
  };

  const std::vector<std::vector<int>> expected_testcase_outputs(
      {{
           // Unigram & Bigram output for "wow", when `max_splits` == 2.
           // We cannot include any of the actual tokens, since `max_splits`
           // only allows enough space for the delimiters.
           hash("^", 0),
           hash("$", 0),
           hash("^$", 1),
           hash("$", 1),
       },
       {
           // Unigram & Bigram output for "wow", when `max_splits` == 3.
           // We can start to include some tokens from the input string.
           hash("^", 0),
           hash("w", 0),
           hash("$", 0),
           hash("^w", 1),
           hash("w$", 1),
           hash("$", 1),
       },
       {
           // Unigram & Bigram output for "wow", when `max_splits` == 4.
           hash("^", 0),
           hash("w", 0),
           hash("o", 0),
           hash("$", 0),
           hash("^w", 1),
           hash("wo", 1),
           hash("o$", 1),
           hash("$", 1),
       },
       {
           // Unigram & Bigram output for "wow", when `max_splits` == 5.
           // We can include the full input string.
           hash("^", 0),
           hash("w", 0),
           hash("o", 0),
           hash("w", 0),
           hash("$", 0),
           hash("^w", 1),
           hash("wo", 1),
           hash("ow", 1),
           hash("w$", 1),
           hash("$", 1),
       },
       {
           // Unigram & Bigram output for "wow", when `max_splits` == 6.
           // `max_splits` is more than the full input string.
           hash("^", 0),
           hash("w", 0),
           hash("o", 0),
           hash("w", 0),
           hash("$", 0),
           hash("^w", 1),
           hash("wo", 1),
           hash("ow", 1),
           hash("w$", 1),
           hash("$", 1),
       }});

  for (int test_idx = 0; test_idx < max_splits.size(); test_idx++) {
    const int testcase_max_splits = max_splits[test_idx];
    NGramHashModel m(kSeed, ngram_lengths, vocab_sizes, testcase_max_splits);
    m.Invoke(testcase_input);
    SCOPED_TRACE(Message() << "Where `max_splits` is: " << testcase_max_splits);
    EXPECT_THAT(m.GetOutput<int>(),
                ElementsAreArray(expected_testcase_outputs[test_idx]));
    EXPECT_THAT(
        m.GetOutputShape(),
        ElementsAreArray(
            {/*batch_size=*/1, static_cast<int>(ngram_lengths.size()),
             std::min(
                 // Longest possible tokenization when using the entire
                 // input.
                 static_cast<int>(testcase_input.size()) + /*padding*/ 2,
                 // Longest possible string when the `max_splits` value
                 // is < testcase_input.size() + 2 for padding.
                 testcase_max_splits)}));
  }
}

TEST(NGramHashTest, InvalidMaxSplitsValue) {
  // Check that the op errors out when given an invalid max splits value.
  const std::vector<int> invalid_max_splits({0, -1, -5, -100});
  for (const int max_splits : invalid_max_splits) {
    NGramHashModel m(/*seed=*/123, /*ngram_lengths=*/{100, 200},
                     /*vocab_sizes=*/{1, 2}, /*max_splits=*/max_splits);
    EXPECT_EQ(m.InvokeUnchecked("hi"), kTfLiteError);
  }
}

TEST(NGramHashTest, MismatchNgramLengthsAndVocabSizes) {
  // Check that the op errors out when ngram lengths and vocab sizes mistmatch.
  {
    NGramHashModel m(/*seed=*/123, /*ngram_lengths=*/{100, 200, 300},
                     /*vocab_sizes=*/{1, 2});
    EXPECT_EQ(m.InvokeUnchecked("hi"), kTfLiteError);
  }
  {
    NGramHashModel m(/*seed=*/123, /*ngram_lengths=*/{100, 200},
                     /*vocab_sizes=*/{1, 2, 3});
    EXPECT_EQ(m.InvokeUnchecked("hi"), kTfLiteError);
  }
}

}  // namespace
}  // namespace mediapipe::tflite_operations

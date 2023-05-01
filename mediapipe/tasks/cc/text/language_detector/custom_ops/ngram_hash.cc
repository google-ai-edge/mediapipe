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
#include <string>
#include <vector>

#include "flatbuffers/flexbuffers.h"
#include "mediapipe/tasks/cc/text/language_detector/custom_ops/utils/hash/murmur.h"
#include "mediapipe/tasks/cc/text/language_detector/custom_ops/utils/ngram_hash_ops_utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"

namespace mediapipe::tflite_operations {

namespace ngram_op {

namespace {

using ::flexbuffers::GetRoot;
using ::flexbuffers::Map;
using ::flexbuffers::TypedVector;
using ::mediapipe::tasks::text::language_detector::custom_ops::
    LowercaseUnicodeStr;
using ::mediapipe::tasks::text::language_detector::custom_ops::Tokenize;
using ::mediapipe::tasks::text::language_detector::custom_ops::TokenizedOutput;
using ::mediapipe::tasks::text::language_detector::custom_ops::hash::
    MurmurHash64WithSeed;
using ::tflite::GetString;
using ::tflite::StringRef;

constexpr int kInputMessage = 0;
constexpr int kOutputLabel = 0;
constexpr int kDefaultMaxSplits = 128;

// This op takes in a string, finds the character ngrams for it and then
// maps each of these ngrams to an index using the specified vocabulary sizes.

// Input(s):
// - input: Input string.
// - seeds: Seed for the random number generator.
// - ngram_lengths: Lengths of each of the ngrams. For example [1, 2, 3] would
//   be interpreted as generating unigrams, bigrams, and trigrams.
// - vocab_sizes: Size of the vocabulary for each of the ngram features
//   respectively. The op would generate vocab ids to be less than or equal to
//   the vocab size. The index 0 implies an invalid ngram.
// - max_splits: Maximum number of tokens in the output. If this is unset, the
//   limit is `kDefaultMaxSplits`.
// - lower_case_input: If this is set to true, the input string would be
//   lower-cased before any processing.

// Output(s):
// - output: A tensor of size [number of ngrams, number of tokens + 2],
//   where 2 tokens are reserved for the padding. If `max_splits` is set, this
//   length is <= max_splits, otherwise it is <= `kDefaultMaxSplits`.

// Helper class used for pre-processing the input.
class NGramHashParams {
 public:
  NGramHashParams(const uint64_t seed, const std::vector<int>& ngram_lengths,
                  const std::vector<int>& vocab_sizes, int max_splits,
                  bool lower_case_input)
      : seed_(seed),
        ngram_lengths_(ngram_lengths),
        vocab_sizes_(vocab_sizes),
        max_splits_(max_splits),
        lower_case_input_(lower_case_input) {}

  TfLiteStatus PreprocessInput(const TfLiteTensor* input_t,
                               TfLiteContext* context) {
    if (input_t->bytes == 0) {
      context->ReportError(context, "Empty input not supported.");
      return kTfLiteError;
    }

    // Do sanity checks on the input.
    if (ngram_lengths_.empty()) {
      context->ReportError(context, "`ngram_lengths` must be non-empty.");
      return kTfLiteError;
    }

    if (vocab_sizes_.empty()) {
      context->ReportError(context, "`vocab_sizes` must be non-empty.");
      return kTfLiteError;
    }

    if (ngram_lengths_.size() != vocab_sizes_.size()) {
      context->ReportError(
          context,
          "Sizes of `ngram_lengths` and `vocab_sizes` must be the same.");
      return kTfLiteError;
    }

    if (max_splits_ <= 0) {
      context->ReportError(context, "`max_splits` must be > 0.");
      return kTfLiteError;
    }

    // Obtain and tokenize the input.
    StringRef inputref = GetString(input_t, /*string_index=*/0);
    if (lower_case_input_) {
      std::string lower_cased_str;
      LowercaseUnicodeStr(inputref.str, inputref.len, &lower_cased_str);

      tokenized_output_ =
          Tokenize(lower_cased_str.c_str(), inputref.len, max_splits_,
                   /*exclude_nonalphaspace_tokens=*/true);
    } else {
      tokenized_output_ = Tokenize(inputref.str, inputref.len, max_splits_,
                                   /*exclude_nonalphaspace_tokens=*/true);
    }
    return kTfLiteOk;
  }
  uint64_t GetSeed() const { return seed_; }

  int GetNumTokens() const { return tokenized_output_.tokens.size(); }

  int GetNumNGrams() const { return ngram_lengths_.size(); }

  std::vector<int> GetNGramLengths() const { return ngram_lengths_; }

  std::vector<int> GetVocabSizes() const { return vocab_sizes_; }

  const TokenizedOutput& GetTokenizedOutput() const {
    return tokenized_output_;
  }

  TokenizedOutput tokenized_output_;

 private:
  const uint64_t seed_;
  std::vector<int> ngram_lengths_;
  std::vector<int> vocab_sizes_;
  const int max_splits_;
  const bool lower_case_input_;
};

// Convert the TypedVector into a regular std::vector.
std::vector<int> GetIntVector(TypedVector typed_vec) {
  std::vector<int> vec(typed_vec.size());
  for (int j = 0; j < typed_vec.size(); j++) {
    vec[j] = typed_vec[j].AsInt32();
  }
  return vec;
}

void GetNGramHashIndices(NGramHashParams* params, int32_t* data) {
  const int max_unicode_length = params->GetNumTokens();
  const auto ngram_lengths = params->GetNGramLengths();
  const auto vocab_sizes = params->GetVocabSizes();
  const auto& tokenized_output = params->GetTokenizedOutput();
  const auto seed = params->GetSeed();

  // Compute for each ngram.
  for (int ngram = 0; ngram < ngram_lengths.size(); ngram++) {
    const int vocab_size = vocab_sizes[ngram];
    const int ngram_length = ngram_lengths[ngram];

    // Compute for each token within the input.
    for (int start = 0; start < tokenized_output.tokens.size(); start++) {
      // Compute the number of bytes for the ngram starting at the given
      // token.
      int num_bytes = 0;
      for (int i = start;
           i < tokenized_output.tokens.size() && i < (start + ngram_length);
           i++) {
        num_bytes += tokenized_output.tokens[i].second;
      }

      // Compute the hash for the ngram starting at the token.
      const auto str_hash = MurmurHash64WithSeed(
          tokenized_output.str.c_str() + tokenized_output.tokens[start].first,
          num_bytes, seed);

      // Map the hash to an index in the vocab.
      data[ngram * max_unicode_length + start] = (str_hash % vocab_size) + 1;
    }
  }
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const Map& m = GetRoot(buffer_t, length).AsMap();

  const uint64_t seed = m["seed"].AsUInt64();
  const std::vector<int> ngram_lengths =
      GetIntVector(m["ngram_lengths"].AsTypedVector());
  const std::vector<int> vocab_sizes =
      GetIntVector(m["vocab_sizes"].AsTypedVector());
  const int max_splits =
      m["max_splits"].IsNull() ? kDefaultMaxSplits : m["max_splits"].AsInt32();
  const bool lowercase_input =
      m["lowercase_input"].IsNull() ? true : m["lowercase_input"].AsBool();

  return new NGramHashParams(seed, ngram_lengths, vocab_sizes, max_splits,
                             lowercase_input);
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<NGramHashParams*>(buffer);
}

TfLiteStatus Resize(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output = tflite::GetOutput(context, node, kOutputLabel);
  TF_LITE_ENSURE(context, output != nullptr);
  tflite::SetTensorToDynamic(output);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  NGramHashParams* params = reinterpret_cast<NGramHashParams*>(node->user_data);
  TF_LITE_ENSURE_OK(
      context, params->PreprocessInput(
                   tflite::GetInput(context, node, kInputMessage), context));

  TfLiteTensor* output = tflite::GetOutput(context, node, kOutputLabel);
  TF_LITE_ENSURE(context, output != nullptr);
  if (tflite::IsDynamicTensor(output)) {
    TfLiteIntArray* output_size = TfLiteIntArrayCreate(3);
    output_size->data[0] = 1;
    output_size->data[1] = params->GetNumNGrams();
    output_size->data[2] = params->GetNumTokens();
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, output, output_size));
  } else {
    context->ReportError(context, "Output must by dynamic.");
    return kTfLiteError;
  }

  if (output->type == kTfLiteInt32) {
    GetNGramHashIndices(params, output->data.i32);
  } else {
    context->ReportError(context, "Output type must be Int32.");
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace ngram_op

TfLiteRegistration* Register_NGRAM_HASH() {
  static TfLiteRegistration r = {ngram_op::Init, ngram_op::Free,
                                 ngram_op::Resize, ngram_op::Eval};
  return &r;
}

}  // namespace mediapipe::tflite_operations

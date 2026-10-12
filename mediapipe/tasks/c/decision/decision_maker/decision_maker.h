/* Copyright 2026 The MediaPipe Authors.

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

#ifndef MEDIAPIPE_TASKS_C_DECISION_DECISION_MAKER_DECISION_MAKER_H_
#define MEDIAPIPE_TASKS_C_DECISION_DECISION_MAKER_DECISION_MAKER_H_

#include <stdbool.h>

#include "mediapipe/tasks/c/core/base_options.h"
#include "mediapipe/tasks/c/core/mp_status.h"

#ifndef MP_EXPORT
#if defined(_MSC_VER)
#define MP_EXPORT __declspec(dllexport)
#else
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // _MSC_VER
#endif  // MP_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

// The opaque pointer to a MediaPipe Decision task.
typedef struct MpDecisionMakerInternal* MpDecisionMakerPtr;

// Multimodal decision context for evaluating text, image, and/or audio state.
struct MpDecisionContext {
  // Optional UTF-8 text context (may be null or empty).
  const char* text;
  // Optional encoded image bytes (PNG, JPEG, WebP, or TGA; may be null).
  const unsigned char* image_bytes;
  // Number of bytes in `image_bytes`.
  int image_bytes_length;
  // Optional mono 16kHz PCM float32 audio samples in [-1.0, 1.0] (may be null).
  const float* audio_samples;
  // Number of float samples in `audio_samples`.
  int audio_samples_length;
};

// Configuration for a Boolean question.
struct MpBooleanQuestion {
  // Natural-language condition statement to evaluate as True or False.
  const char* condition;
  // Probability threshold in [0, 1] above which `value` is True.
  float threshold;
  // Softmax temperature override (0 uses the default temperature).
  float temperature;
  // Whether to subtract the empty-context prior log-odds before thresholding.
  bool normalize_prior;
};

// Configuration for a Choice question.
struct MpChoiceQuestion {
  // Array of `count` option key strings.
  const char* const* keys;
  // Optional array of `count` option description strings (may be null).
  const char* const* descriptions;
  // Number of choices in `keys` (and `descriptions` if non-null).
  int count;
  // Softmax temperature override (0 uses the default temperature).
  float temperature;
  // Optional task instructions prepended to the question prompt.
  const char* instructions;
  // Scoring mode enum value corresponding to `proto::ScoringMode`.
  int scoring_mode;
  // Whether to normalize choice probabilities against the empty-context prior.
  bool normalize_prior;
  // Optional array of `count` encoded image byte buffers (PNG/JPEG/TGA) for
  // multimodal options (may be null; individual entries may be null).
  const unsigned char* const* image_bytes;
  // Optional array of `count` byte lengths corresponding to `image_bytes`.
  const int* image_bytes_lengths;
  // Optional array of `count` mono 16kHz PCM float32 audio buffers for
  // multimodal options (may be null; individual entries may be null).
  const float* const* audio_samples;
  // Optional array of `count` sample counts corresponding to `audio_samples`.
  const int* audio_samples_lengths;
};

// Configuration for a Score question.
struct MpScoreQuestion {
  // Ordered array of `count` rubric level description strings.
  const char* const* rubric;
  // Number of rubric levels in `rubric`.
  int count;
  // Softmax temperature override (0 uses the default temperature).
  float temperature;
  // Optional task instructions prepended to the rubric prompt.
  const char* instructions;
};

// Result of a Boolean evaluation.
struct MpBooleanResult {
  // Whether `probability_true` meets or exceeds the configured threshold.
  bool value;
  // Calibrated probability in [0, 1] that the condition holds.
  float probability_true;
  // Confidence score in [0, 1] for the boolean decision.
  float confidence;
};

// Result of a Choice evaluation.
struct MpChoiceResult {
  // Heap-allocated string key of the highest-probability choice.
  char* selected_key;
  // Heap-allocated array of `count` choice key strings.
  char** keys;
  // Heap-allocated array of `count` probabilities corresponding to `keys`.
  float* probabilities;
  // Number of choice entries in `keys` and `probabilities`.
  int count;
  // Confidence score in [0, 1] for the selected choice.
  float confidence;
};

// Result of a Score evaluation.
struct MpScoreResult {
  // Heap-allocated string key of the highest-probability rubric level.
  char* selected_key;
  // Continuous expected rubric level: sum_i(i * probabilities[i]), in
  // [0, count - 1]. Not normalized to [0, 1].
  float expected_score;
  // Heap-allocated array of `count` rubric level probabilities.
  float* probabilities;
  // Number of rubric levels in `probabilities`.
  int count;
  // Confidence score in [0, 1] for the score evaluation.
  float confidence;
};

// Question type tag for multi-question schema evaluation.
enum MpQuestionType {
  kMpQuestionTypeBoolean = 0,
  kMpQuestionTypeChoice = 1,
  kMpQuestionTypeScore = 2,
};

// Named question entry in a multi-question schema.
struct MpSchemaQuestion {
  // Unique question identifier within the schema.
  const char* name;
  // Question type tag (`kMpQuestionTypeBoolean`, `kMpQuestionTypeChoice`, or
  // `kMpQuestionTypeScore`).
  int question_type;
  // Question payload corresponding to `question_type`.
  struct MpBooleanQuestion boolean_question;
  struct MpChoiceQuestion choice_question;
  struct MpScoreQuestion score_question;
};

// Result entry for a single question in a multi-question schema evaluation.
struct MpSchemaQuestionResult {
  // Question type tag (`kMpQuestionTypeBoolean`, `kMpQuestionTypeChoice`, or
  // `kMpQuestionTypeScore`).
  int question_type;
  // Result payload corresponding to `question_type`.
  struct MpBooleanResult boolean_result;
  struct MpChoiceResult choice_result;
  struct MpScoreResult score_result;
};

// Options for configuring a MediaPipe Decision task.
struct MpDecisionMakerOptions {
  // Common MediaPipe task base options (model asset path/buffer and delegate).
  struct MpBaseOptions base_options;
  // Maximum token budget for sequence construction (0 uses model default).
  int max_num_tokens;
};

// Creates a Decision task from options.
MP_EXPORT MpStatus
MpDecisionMakerCreate(const struct MpDecisionMakerOptions* options,
                      MpDecisionMakerPtr* maker, char** error_msg);

// Evaluates a Boolean condition against input text.
MP_EXPORT MpStatus MpDecisionMakerEvaluateBoolean(
    MpDecisionMakerPtr maker, const char* text,
    const struct MpBooleanQuestion* question, struct MpBooleanResult* result,
    char** error_msg);

// Evaluates a Choice question against input text.
MP_EXPORT MpStatus
MpDecisionMakerEvaluateChoice(MpDecisionMakerPtr maker, const char* text,
                              const struct MpChoiceQuestion* question,
                              struct MpChoiceResult* result, char** error_msg);

// Evaluates a Score rubric against input text.
MP_EXPORT MpStatus
MpDecisionMakerEvaluateScore(MpDecisionMakerPtr maker, const char* text,
                             const struct MpScoreQuestion* question,
                             struct MpScoreResult* result, char** error_msg);

// Evaluates a multi-question schema simultaneously against input text in a
// single forward pass when supported by the backend.
MP_EXPORT MpStatus MpDecisionMakerEvaluateSchema(
    MpDecisionMakerPtr maker, const char* text,
    const struct MpSchemaQuestion* questions, int num_questions,
    struct MpSchemaQuestionResult* results, char** error_msg);

// Evaluates a text-only JSON decision request (MediaPipe Decision or OpenAI
// `json_schema` format) and returns a heap-allocated JSON response string in
// `*json_result` (free with `MpDecisionMakerFreeJson`).
MP_EXPORT MpStatus MpDecisionMakerEvaluateJson(MpDecisionMakerPtr maker,
                                               const char* json_request,
                                               char** json_result,
                                               char** error_msg);

// Evaluates a JSON question schema against a zero-copy multimodal
// `MpDecisionContext` and returns a heap-allocated JSON response string in
// `*json_result` (free with `MpDecisionMakerFreeJson`).
MP_EXPORT MpStatus MpDecisionMakerEvaluateContextJson(
    MpDecisionMakerPtr maker, const struct MpDecisionContext* context,
    const char* json_questions, char** json_result, char** error_msg);

// Frees a heap-allocated JSON result string returned by
// `MpDecisionMakerEvaluateJson` or `MpDecisionMakerEvaluateContextJson`.
MP_EXPORT void MpDecisionMakerFreeJson(char* json_result);

// Evaluates a Boolean condition against a multimodal decision context.
MP_EXPORT MpStatus MpDecisionMakerEvaluateContextBoolean(
    MpDecisionMakerPtr maker, const struct MpDecisionContext* context,
    const struct MpBooleanQuestion* question, struct MpBooleanResult* result,
    char** error_msg);

// Evaluates a Choice question against a multimodal decision context.
MP_EXPORT MpStatus MpDecisionMakerEvaluateContextChoice(
    MpDecisionMakerPtr maker, const struct MpDecisionContext* context,
    const struct MpChoiceQuestion* question, struct MpChoiceResult* result,
    char** error_msg);

// Evaluates a Score rubric against a multimodal decision context.
MP_EXPORT MpStatus MpDecisionMakerEvaluateContextScore(
    MpDecisionMakerPtr maker, const struct MpDecisionContext* context,
    const struct MpScoreQuestion* question, struct MpScoreResult* result,
    char** error_msg);

// Evaluates a multi-question schema against a multimodal decision context.
MP_EXPORT MpStatus MpDecisionMakerEvaluateContextSchema(
    MpDecisionMakerPtr maker, const struct MpDecisionContext* context,
    const struct MpSchemaQuestion* questions, int num_questions,
    struct MpSchemaQuestionResult* results, char** error_msg);

// Evaluates a Boolean condition against a batch of input texts
// (with an optional shared_prefix_text for candidate batching).
MP_EXPORT MpStatus MpDecisionMakerEvaluateBooleanBatch(
    MpDecisionMakerPtr maker, const char* shared_prefix_text,
    const char* const* texts, int num_texts,
    const struct MpBooleanQuestion* question, struct MpBooleanResult* results,
    char** error_msg);

// Evaluates a Choice question against a batch of input texts
// (with an optional shared_prefix_text for candidate batching).
MP_EXPORT MpStatus MpDecisionMakerEvaluateChoiceBatch(
    MpDecisionMakerPtr maker, const char* shared_prefix_text,
    const char* const* texts, int num_texts,
    const struct MpChoiceQuestion* question, struct MpChoiceResult* results,
    char** error_msg);

// Evaluates a Score rubric against a batch of input texts
// (with an optional shared_prefix_text for candidate batching).
MP_EXPORT MpStatus MpDecisionMakerEvaluateScoreBatch(
    MpDecisionMakerPtr maker, const char* shared_prefix_text,
    const char* const* texts, int num_texts,
    const struct MpScoreQuestion* question, struct MpScoreResult* results,
    char** error_msg);

// Frees memory allocated within an MpChoiceResult.
MP_EXPORT void MpDecisionMakerCloseChoiceResult(struct MpChoiceResult* result);

// Frees memory allocated within each element of an MpChoiceResult
// batch array of length `count`.
MP_EXPORT void MpDecisionMakerCloseChoiceResultBatch(
    struct MpChoiceResult* results, int count);

// Frees memory allocated within an MpScoreResult.
MP_EXPORT void MpDecisionMakerCloseScoreResult(struct MpScoreResult* result);

// Frees memory allocated within each element of an MpScoreResult
// batch array of length `count`.
MP_EXPORT void MpDecisionMakerCloseScoreResultBatch(
    struct MpScoreResult* results, int count);

// Frees memory allocated within each element of an MpSchemaQuestionResult
// array of length `num_questions`.
MP_EXPORT void MpDecisionMakerCloseSchemaResult(
    struct MpSchemaQuestionResult* results, int num_questions);

// Prewarms the prompt prefix and KV cache for a Choice question.
MP_EXPORT MpStatus MpDecisionMakerPrewarmChoice(
    MpDecisionMakerPtr maker, const struct MpChoiceQuestion* question,
    char** error_msg);

// Prewarms the prompt prefix and KV cache for a multi-question schema.
MP_EXPORT MpStatus MpDecisionMakerPrewarmSchema(
    MpDecisionMakerPtr maker, const struct MpSchemaQuestion* questions,
    int num_questions, char** error_msg);

// Prewarms the prompt prefix and KV cache for a JSON question schema.
MP_EXPORT MpStatus MpDecisionMakerPrewarmJson(MpDecisionMakerPtr maker,
                                              const char* json_schema,
                                              char** error_msg);

// Closes and releases the Decision task instance.
MP_EXPORT MpStatus MpDecisionMakerClose(MpDecisionMakerPtr maker,
                                        char** error_msg);

#ifdef __cplusplus
}
#endif

#endif  // MEDIAPIPE_TASKS_C_DECISION_DECISION_MAKER_DECISION_MAKER_H_

#ifndef MEDIAPIPE_TASKS_GENAI_INFERENCE_UTILS_LLM_UTILS_TOKENIZER_UTILS_H_
#define MEDIAPIPE_TASKS_GENAI_INFERENCE_UTILS_LLM_UTILS_TOKENIZER_UTILS_H_

#include <string>

namespace mediapipe::tasks::genai::llm_utils {

// Returns a serialized proto with the targeted vocab size that can be used to
// initialize a SentencePieceProcessor.
std::string GetFakeSerializedVocabProto(int vocab_size);

}  // namespace mediapipe::tasks::genai::llm_utils

#endif  // MEDIAPIPE_TASKS_GENAI_INFERENCE_UTILS_LLM_UTILS_TOKENIZER_UTILS_H_

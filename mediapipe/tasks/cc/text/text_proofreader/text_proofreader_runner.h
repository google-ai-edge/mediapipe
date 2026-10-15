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

#ifndef MEDIAPIPE_TASKS_CC_TEXT_TEXT_PROOFREADER_TEXT_PROOFREADER_RUNNER_H_
#define MEDIAPIPE_TASKS_CC_TEXT_TEXT_PROOFREADER_TEXT_PROOFREADER_RUNNER_H_

#include <memory>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "c/conversation.h"  // from @litert_lm
#include "c/engine.h"        // from @litert_lm
#include "mediapipe/tasks/cc/text/text_proofreader/text_proofreader.h"

namespace mediapipe::tasks::text::text_proofreader {

namespace internal_text_proofreader {

// A custom deleter for LiteRtLmEngine to be used with std::unique_ptr.
struct LiteRtLmEngineDeleter {
  void operator()(LiteRtLmEngine* engine) const {
    if (engine) {
      litert_lm_engine_delete(engine);
    }
  }
};

// A custom deleter for LiteRtLmConversation to be used with std::unique_ptr.
struct LiteRtLmConversationDeleter {
  void operator()(LiteRtLmConversation* conversation) const {
    if (conversation) {
      litert_lm_conversation_delete(conversation);
    }
  }
};

}  // namespace internal_text_proofreader

// Internal runner class that handles direct interaction with the LiteRT LM
// engine.
// In this first version, it acts as a pass-through (no chunking).
class TextProofreaderRunner {
 public:
  static absl::StatusOr<std::unique_ptr<TextProofreaderRunner>> Create(
      std::unique_ptr<TextProofreaderOptions> options);

  ~TextProofreaderRunner();

  // Runs the model on the input text streaming the results.
  // The callback receives the partial text chunk and a boolean indicating if
  // it is the final chunk.
  absl::Status RunStreaming(
      absl::string_view text,
      absl::AnyInvocable<void(absl::StatusOr<absl::string_view>, bool)>
          result_callback) ABSL_LOCKS_EXCLUDED(mutex_);

  void Cancel() ABSL_LOCKS_EXCLUDED(mutex_);

  absl::Status Close();

 private:
  using LiteRtLmEnginePtr =
      std::unique_ptr<LiteRtLmEngine,
                      internal_text_proofreader::LiteRtLmEngineDeleter>;
  using LiteRtLmConversationPtr =
      std::unique_ptr<LiteRtLmConversation,
                      internal_text_proofreader::LiteRtLmConversationDeleter>;

  explicit TextProofreaderRunner(LiteRtLmEnginePtr engine, int max_num_tokens);

  absl::StatusOr<LiteRtLmConversationPtr> CreateConversation(
      absl::string_view text);

  absl::StatusOr<LiteRtLmConversation*> ResetAndCreateConversation(
      absl::string_view text) ABSL_LOCKS_EXCLUDED(mutex_);

  struct StreamContext;

  static void StaticStreamCallback(void* callback_data,
                                   const LiteRtLmStreamChunk* chunk);

  void HandleStreamCallback(StreamContext* context, absl::string_view chunk,
                            bool is_final, absl::string_view error_msg);

  void DeliverErrorAndCleanup(StreamContext* context, absl::Status status);

  LiteRtLmEnginePtr engine_;
  int max_num_tokens_;

  absl::Mutex mutex_;
  LiteRtLmConversationPtr conversation_ ABSL_GUARDED_BY(mutex_);
  bool cancelled_ ABSL_GUARDED_BY(mutex_) = false;
};

}  // namespace mediapipe::tasks::text::text_proofreader

#endif  // MEDIAPIPE_TASKS_CC_TEXT_TEXT_PROOFREADER_TEXT_PROOFREADER_RUNNER_H_

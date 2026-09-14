#ifdef __EMSCRIPTEN__

#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/em_js.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/tasks/cc/text/text_summarizer/text_summarizer.h"

namespace mediapipe {

using ::mediapipe::tasks::text::text_summarizer::TextSummarizer;
using ::mediapipe::tasks::text::text_summarizer::TextSummarizerOptions;

namespace {

EM_JS(void, TextSummarizerThrowJsError, (const char* message),
      { throw new Error(UTF8ToString(message)); });

void ReportError(const std::string& msg) {
  TextSummarizerThrowJsError(msg.c_str());
}

// Default absl logging reports all log levels (including INFO and WARNING) to
// stderr, which Emscripten routes to console.error in the browser. We silence
// non-error logging for this WASM build to prevent polluting the browser
// console with internal engine logs, so logging is intentionally not
// re-enabled.
void DisableLogging() {
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kError);
}

}  // namespace

intptr_t CreateTextSummarizer(std::string model_path, int max_num_tokens,
                              std::string mode) {
  DisableLogging();

  auto options = std::make_unique<TextSummarizerOptions>();
  options->base_options.model_asset_path = std::move(model_path);
  if (max_num_tokens > 0) {
    options->max_num_tokens = max_num_tokens;
  }

  if (mode == "TLDR") {
    options->mode = TextSummarizerOptions::Mode::TLDR;
  } else if (mode == "KEYPOINTS") {
    options->mode = TextSummarizerOptions::Mode::KEYPOINTS;
  } else {
    ReportError(absl::StrCat("Unsupported mode: ", mode));
    return 0;
  }

  options->cache_dir = ":nocache";

  options->parallel_file_section_loading = false;

  auto result = TextSummarizer::Create(std::move(options));
  if (!result.ok()) {
    ReportError(absl::StrCat("Failed to create TextSummarizer: ",
                             result.status().message()));
    return 0;
  }
  return reinterpret_cast<intptr_t>(result.value().release());
}

void DeleteTextSummarizer(intptr_t summarizer_ptr) {
  DisableLogging();
  auto* summarizer = reinterpret_cast<TextSummarizer*>(summarizer_ptr);
  if (summarizer) {
    summarizer->Close().IgnoreError();
    delete summarizer;
  }
}

std::string Summarize(intptr_t summarizer_ptr, std::string text) {
  DisableLogging();
  auto* summarizer = reinterpret_cast<TextSummarizer*>(summarizer_ptr);
  if (!summarizer) {
    ReportError("TextSummarizer has not been created!");
    return "";
  }

  auto result = summarizer->Summarize(text);
  if (!result.ok()) {
    ReportError(
        absl::StrCat("Failed to summarize: ", result.status().message()));
    return "";
  }
  return std::move(result->summary);
}

EMSCRIPTEN_BINDINGS(text_summarizer_api) {
  emscripten::function("createTextSummarizer", &CreateTextSummarizer);
  emscripten::function("deleteTextSummarizer", &DeleteTextSummarizer);
  emscripten::function("summarize", &Summarize);
}

}  // namespace mediapipe

#endif  // __EMSCRIPTEN__

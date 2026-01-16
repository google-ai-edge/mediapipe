/* Copyright 2025 The MediaPipe Authors.

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

#include "mediapipe/tasks/c/metadata/flatbuffer_api.h"

#include <cstdint>
#include <cstdlib>
#include <string>

#include "absl/status/status.h"
#include "flatbuffers/idl.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/core/mp_status_converter.h"

extern "C" {

struct MpFlatbufferParserInternal {
  flatbuffers::Parser parser;
};

MpStatus MpFlatbufferParserCreate(bool enable_strict_json,
                                  MpFlatbufferParser* parser_out,
                                  char** error_message) {
  if (!parser_out) {
    return mediapipe::tasks::c::core::HandleStatus(
        absl::InvalidArgumentError("parser_out is null"), error_message);
  }
  flatbuffers::IDLOptions opts;
  opts.strict_json = enable_strict_json;
  *parser_out =
      new MpFlatbufferParserInternal{.parser = flatbuffers::Parser(opts)};
  return kMpOk;
}

MpStatus MpFlatbufferParserParse(MpFlatbufferParser parser, const char* source,
                                 char** error_message) {
  if (!parser->parser.Parse(source)) {
    return mediapipe::tasks::c::core::HandleStatus(
        absl::InvalidArgumentError(parser->parser.error_), error_message);
  }
  return kMpOk;
}

const char* MpFlatbufferParserGetError(MpFlatbufferParser parser) {
  return parser->parser.error_.c_str();
}

MpStatus MpFlatbufferGenerateText(MpFlatbufferParser parser,
                                  const uint8_t* buffer, size_t buffer_size,
                                  char** json_out, char** error_message) {
  std::string text;
  const char* error_str = flatbuffers::GenText(
      parser->parser, reinterpret_cast<const void*>(buffer), &text);
  if (error_str) {
    *json_out = nullptr;
    return mediapipe::tasks::c::core::HandleStatus(
        absl::InternalError(error_str), error_message);
  }
  *json_out = strdup(text.c_str());
  return kMpOk;
}

void MpFlatbufferFreeString(char* str) { free(str); }

void MpFlatbufferParserDelete(MpFlatbufferParser parser) {
  if (parser) {
    delete parser;
  }
}

}  // extern "C"

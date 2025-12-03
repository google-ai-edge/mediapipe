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

#ifndef MEDIAPIPE_TASKS_C_METADATA_FLATBUFFER_API_H_
#define MEDIAPIPE_TASKS_C_METADATA_FLATBUFFER_API_H_

#include <stdbool.h>
#include <stdint.h>

#include <cstddef>

#include "mediapipe/tasks/c/core/mp_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MP_EXPORT
#define MP_EXPORT __attribute__((visibility("default")))
#endif  // MP_EXPORT

// Represents a Flatbuffer Parser.
typedef struct MpFlatbufferParserInternal* MpFlatbufferParser;

// Creates a new Flatbuffer Parser.
// Returns: kMpOk on success, otherwise an error code.
MP_EXPORT MpStatus MpFlatbufferParserCreate(bool enable_strict_json,
                                            MpFlatbufferParser* parser_out);

// Parses the Flatbuffer schema source.
// Returns: kMpOk on success, otherwise an error code.
MP_EXPORT MpStatus MpFlatbufferParserParse(MpFlatbufferParser parser,
                                           const char* source);

// Gets the error message from the parser.
// The returned string is owned by the Parser and should not be freed.
MP_EXPORT const char* MpFlatbufferParserGetError(MpFlatbufferParser parser);

// Generates JSON text from a Flatbuffer buffer.
// The caller is responsible for freeing the returned string using
// MpFlatbufferFreeString.
MP_EXPORT MpStatus MpFlatbufferGenerateText(MpFlatbufferParser parser,
                                            const uint8_t* buffer,
                                            size_t buffer_size,
                                            char** json_out);

// Frees a string allocated by MpFlatbufferGenerateText.
MP_EXPORT void MpFlatbufferFreeString(char* str);

// Deletes a Flatbuffer Parser.
MP_EXPORT void MpFlatbufferParserDelete(MpFlatbufferParser parser);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MEDIAPIPE_TASKS_C_METADATA_FLATBUFFER_API_H_

/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

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

#include "mediapipe/tasks/cc/retrieval/semantic_retriever/audio_decoder.h"

#include <string.h>

#include <cinttypes>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/port/status_macros.h"

namespace mediapipe::tasks::retrieval {
namespace {

constexpr char kRiffChunkId[] = "RIFF";
constexpr char kRiffType[] = "WAVE";
constexpr char kFormatChunkId[] = "fmt ";
constexpr char kDataChunkId[] = "data";

inline float Int16SampleToFloat(int16_t data) {
  constexpr float kMultiplier = 1.0f / (1 << 15);
  return data * kMultiplier;
}

namespace port {
constexpr bool kLittleEndian = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;
}  // namespace port

// Handles moving the data index forward, validating the arguments, and avoiding
// overflow or underflow.
absl::Status IncrementOffset(uint32_t old_offset, size_t increment,
                             size_t max_size, uint32_t* new_offset) {
  if (old_offset > max_size) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Initial offset is outside data range: %d", old_offset));
  }
  const uint64_t sum = static_cast<uint64_t>(old_offset) + increment;
  if (sum > max_size) {
    return absl::InvalidArgumentError(
        "Data too short when trying to read string");
  }
  if (sum > std::numeric_limits<uint32_t>::max()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Offset too large, overflowed: %llu", sum));
  }
  *new_offset = static_cast<uint32_t>(sum);
  return absl::OkStatus();
}

template <class T>
absl::Status ReadValue(absl::string_view data, T* value, uint32_t* offset) {
  uint32_t new_offset;
  ABSL_RETURN_IF_ERROR(
      IncrementOffset(*offset, sizeof(T), data.size(), &new_offset));
  if (port::kLittleEndian) {
    memcpy(value, data.data() + *offset, sizeof(T));
  } else {
    *value = 0;
    const uint8_t* data_buf =
        reinterpret_cast<const uint8_t*>(data.data() + *offset);
    int shift = 0;
    for (int i = 0; i < sizeof(T); ++i, shift += 8) {
      *value = *value | (data_buf[i] << shift);
    }
  }
  *offset = new_offset;
  return absl::OkStatus();
}

absl::Status ExpectText(absl::string_view data, absl::string_view expected_text,
                        uint32_t* offset) {
  uint32_t new_offset;
  ABSL_RETURN_IF_ERROR(
      IncrementOffset(*offset, expected_text.size(), data.size(), &new_offset));
  const absl::string_view found_text =
      data.substr(*offset, new_offset - *offset);
  if (found_text != expected_text) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Header mismatch: Expected", expected_text, " but found ", found_text));
  }
  *offset = new_offset;
  return absl::OkStatus();
}

absl::Status ReadString(absl::string_view data, size_t expected_length,
                        std::string* value, uint32_t* offset) {
  uint32_t new_offset;
  ABSL_RETURN_IF_ERROR(
      IncrementOffset(*offset, expected_length, data.size(), &new_offset));
  *value = std::string(data.substr(*offset, new_offset - *offset));
  *offset = new_offset;
  return absl::OkStatus();
}

absl::Status DecodeLin16WaveAsFloatVector(absl::string_view wav_string,
                                          std::vector<float>* float_values,
                                          uint32_t* offset,
                                          uint32_t* sample_count,
                                          uint16_t* channel_count,
                                          uint32_t* sample_rate) {
  ABSL_RETURN_IF_ERROR(ExpectText(wav_string, kRiffChunkId, offset));
  uint32_t total_file_size;
  ABSL_RETURN_IF_ERROR(
      ReadValue<uint32_t>(wav_string, &total_file_size, offset));
  ABSL_RETURN_IF_ERROR(ExpectText(wav_string, kRiffType, offset));
  ABSL_RETURN_IF_ERROR(ExpectText(wav_string, kFormatChunkId, offset));
  uint32_t format_chunk_size;
  ABSL_RETURN_IF_ERROR(
      ReadValue<uint32_t>(wav_string, &format_chunk_size, offset));
  if ((format_chunk_size != 16) && (format_chunk_size != 18)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Bad format chunk size for WAV: Expected 16 or 18, but got %" PRIu32,
        format_chunk_size));
  }
  uint16_t audio_format;
  ABSL_RETURN_IF_ERROR(ReadValue<uint16_t>(wav_string, &audio_format, offset));
  if (audio_format != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Bad audio format for WAV: Expected 1 (PCM), but got %" PRIu16,
        audio_format));
  }
  ABSL_RETURN_IF_ERROR(ReadValue<uint16_t>(wav_string, channel_count, offset));
  if (*channel_count < 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Bad number of channels for WAV: Expected at least 1, but got %" PRIu16,
        *channel_count));
  }
  ABSL_RETURN_IF_ERROR(ReadValue<uint32_t>(wav_string, sample_rate, offset));
  uint32_t bytes_per_second;
  ABSL_RETURN_IF_ERROR(
      ReadValue<uint32_t>(wav_string, &bytes_per_second, offset));
  uint16_t bytes_per_sample;
  ABSL_RETURN_IF_ERROR(
      ReadValue<uint16_t>(wav_string, &bytes_per_sample, offset));
  // Confusingly, bits per sample is defined as holding the number of bits for
  // one channel, unlike the definition of sample used elsewhere in the WAV
  // spec. For example, bytes per sample is the memory needed for all channels
  // for one point in time.
  uint16_t bits_per_sample;
  ABSL_RETURN_IF_ERROR(
      ReadValue<uint16_t>(wav_string, &bits_per_sample, offset));
  if (bits_per_sample != 16) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Can only read 16-bit WAV files, but received %" PRIu16,
                        bits_per_sample));
  }
  const uint32_t expected_bytes_per_sample =
      ((bits_per_sample * *channel_count) + 7) / 8;
  if (bytes_per_sample != expected_bytes_per_sample) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Bad bytes per sample in WAV header: Expected %" PRIu32
                        " but got %" PRIu16,
                        expected_bytes_per_sample, bytes_per_sample));
  }
  const uint32_t expected_bytes_per_second = bytes_per_sample * *sample_rate;
  if (bytes_per_second != expected_bytes_per_second) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Bad bytes per second in WAV header: Expected %" PRIu32
                        " but got %" PRIu32 " (sample_rate=%" PRIu32
                        ", bytes_per_sample=%" PRIu16 ")",
                        expected_bytes_per_second, bytes_per_second,
                        *sample_rate, bytes_per_sample));
  }
  if (format_chunk_size == 18) {
    // Skip over this unused section.
    *offset += 2;
  }

  bool was_data_found = false;
  while (*offset < wav_string.size()) {
    std::string chunk_id;
    ABSL_RETURN_IF_ERROR(ReadString(wav_string, 4, &chunk_id, offset));
    uint32_t chunk_size;
    ABSL_RETURN_IF_ERROR(ReadValue<uint32_t>(wav_string, &chunk_size, offset));
    if (chunk_size > std::numeric_limits<int32_t>::max()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "WAV data chunk '%s' is too large: %" PRIu32
          " bytes, but the limit is %d",
          chunk_id.c_str(), chunk_size, std::numeric_limits<int32_t>::max()));
    }
    if (chunk_id == kDataChunkId) {
      if (was_data_found) {
        return absl::InvalidArgumentError(
            "More than one data chunk found in WAV");
      }
      was_data_found = true;
      *sample_count = chunk_size / bytes_per_sample;
      const uint32_t data_count = *sample_count * *channel_count;
      uint32_t unused_new_offset = 0;
      // Validate that the data exists before allocating space for it
      // (prevent easy OOM errors).
      ABSL_RETURN_IF_ERROR(
          IncrementOffset(*offset, sizeof(int16_t) * data_count,
                          wav_string.size(), &unused_new_offset));
      float_values->resize(data_count);
      for (int i = 0; i < data_count; ++i) {
        int16_t single_channel_value = 0;
        ABSL_RETURN_IF_ERROR(
            ReadValue<int16_t>(wav_string, &single_channel_value, offset));
        (*float_values)[i] = Int16SampleToFloat(single_channel_value);
      }
    } else {
      uint32_t new_offset = 0;
      ABSL_RETURN_IF_ERROR(
          IncrementOffset(*offset, chunk_size, wav_string.size(), &new_offset));
      *offset = new_offset;
    }
  }
  if (!was_data_found) {
    return absl::InvalidArgumentError("No data chunk found in WAV");
  }
  return absl::OkStatus();
}

std::string ReadFile(const std::string filepath) {
  std::ifstream fs(filepath, std::ios::binary);
  if (!fs.is_open()) {
    return "";
  }
  std::string contents((std::istreambuf_iterator<char>(fs)),
                       (std::istreambuf_iterator<char>()));
  return contents;
}

}  // namespace

absl::StatusOr<std::vector<float>> AudioDecoder::DecodeAudioData(
    absl::string_view path) {
  const std::string wav_string = ReadFile(std::string(path));
  if (wav_string.empty()) {
    return absl::NotFoundError(
        absl::StrCat("Failed to read audio file: ", path));
  }
  return DecodeAudioBytes(wav_string);
}

absl::StatusOr<std::vector<float>> AudioDecoder::DecodeAudioBytes(
    absl::string_view wav_bytes, uint32_t* sample_rate) {
  if (wav_bytes.empty()) {
    return absl::InvalidArgumentError("Audio bytes cannot be empty.");
  }

  std::vector<float> decoded_values;
  uint32_t offset = 0;
  uint32_t sample_count = 0;
  uint16_t channel_count = 0;
  uint32_t parsed_sample_rate = 0;

  ABSL_RETURN_IF_ERROR(DecodeLin16WaveAsFloatVector(
      wav_bytes, &decoded_values, &offset, &sample_count, &channel_count,
      &parsed_sample_rate));

  if (channel_count != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Only mono (1 channel) audio is supported, but got: ", channel_count));
  }

  if (sample_rate != nullptr) {
    *sample_rate = parsed_sample_rate;
  }
  return decoded_values;
}

}  // namespace mediapipe::tasks::retrieval

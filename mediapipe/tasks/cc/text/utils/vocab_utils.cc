/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

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

#include "mediapipe/tasks/cc/text/utils/vocab_utils.h"

#include <fstream>

#include "absl/strings/str_split.h"

namespace mediapipe {
namespace tasks {
namespace text {
namespace {

struct membuf : std::streambuf {
  membuf(char* begin, char* end) { this->setg(begin, begin, end); }
};

void ReadIStreamLineByLine(
    std::istream* istream,
    const std::function<void(std::string)>& line_processor) {
  std::string str;
  while (std::getline(*istream, str)) {
    if (!str.empty()) {
      line_processor(str);
    }
  }
}

absl::node_hash_map<std::string, int> ReadIStreamLineSplits(
    std::istream* istream) {
  absl::node_hash_map<std::string, int> vocab_index_map;
  std::string str;
  ReadIStreamLineByLine(istream, [&vocab_index_map](const std::string& str) {
    std::vector<std::string> v = absl::StrSplit(str, ' ');
    vocab_index_map[v[0]] = std::stoi(v[1]);
  });
  return vocab_index_map;
}

std::vector<std::string> ReadIStreamByLine(std::istream* istream) {
  std::vector<std::string> vocab_from_file;
  std::string str;

  ReadIStreamLineByLine(istream, [&vocab_from_file](const std::string& str) {
    vocab_from_file.push_back(str);
  });
  return vocab_from_file;
}

}  // namespace

std::vector<std::string> LoadVocabFromFile(const std::string& path_to_vocab) {
  std::vector<std::string> vocab_from_file;
  std::ifstream in(path_to_vocab.c_str());
  return ReadIStreamByLine(&in);
}

std::vector<std::string> LoadVocabFromBuffer(const char* vocab_buffer_data,
                                             const size_t vocab_buffer_size) {
  membuf sbuf(const_cast<char*>(vocab_buffer_data),
              const_cast<char*>(vocab_buffer_data + vocab_buffer_size));
  std::istream in(&sbuf);
  return ReadIStreamByLine(&in);
}

absl::node_hash_map<std::string, int> LoadVocabAndIndexFromFile(
    const std::string& path_to_vocab) {
  absl::node_hash_map<std::string, int> vocab_index_map;
  std::ifstream in(path_to_vocab.c_str());
  return ReadIStreamLineSplits(&in);
}

absl::node_hash_map<std::string, int> LoadVocabAndIndexFromBuffer(
    const char* vocab_buffer_data, const size_t vocab_buffer_size) {
  membuf sbuf(const_cast<char*>(vocab_buffer_data),
              const_cast<char*>(vocab_buffer_data + vocab_buffer_size));
  absl::node_hash_map<std::string, int> vocab_index_map;
  std::istream in(&sbuf);
  return ReadIStreamLineSplits(&in);
}

}  // namespace text
}  // namespace tasks
}  // namespace mediapipe

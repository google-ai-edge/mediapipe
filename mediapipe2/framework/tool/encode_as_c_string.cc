// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// This program takes one input file and encodes its contents as a C++
// std::string, which can be included in a C++ source file. It is similar to
// filewrapper (and borrows some of its code), but simpler.

#include <fstream>
#include <iostream>
#include <memory>

#include "absl/strings/escaping.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: encode_as_c_string input_file\n";
    return 1;
  }
  const std::string input_name = argv[1];
  std::ifstream input_file(input_name,
                           std::ios_base::in | std::ios_base::binary);
  if (!input_file.is_open()) {
    std::cerr << "cannot open '" << input_name << "'\n";
    return 2;
  }
  constexpr int kBufSize = 4096;
  std::unique_ptr<char[]> buf(new char[kBufSize]);
  std::cout << "\"";
  int line_len = 1;
  while (1) {
    input_file.read(buf.get(), kBufSize);
    int count = input_file.gcount();
    if (count == 0) break;
    for (int i = 0; i < count; ++i) {
      std::string out = absl::CEscape(absl::string_view(&buf[i], 1));
      if (line_len + out.size() > 79) {
        std::cout << "\"\n\"";
        line_len = 1;
      }
      std::cout << out;
      line_len += out.size();
    }
  }
  input_file.close();
  if (!input_file.eof()) {
    std::cerr << "error reading '" << input_name << "'\n";
    return 2;
  }
  std::cout << "\"\n";
  return 0;
}

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

#include <algorithm>
#include <fstream>

#include "absl/container/btree_map.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "mediapipe/framework/port/advanced_proto_inc.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/profiler/reporter/reporter.h"

ABSL_FLAG(std::vector<std::string>, logfiles, {},
          "comma-separated list of .binarypb files to process.");
ABSL_FLAG(std::vector<std::string>, cols, {"*"},
          "comma-separated list of columns to show. Suffix wildcards, '*', '?' "
          "allowed.");
ABSL_FLAG(bool, compact, false,
          "if true, then don't print unnecessary whitespace.");

using ::mediapipe::reporter::Reporter;

// The command line utility to mine trace files of useful statistics to
// determine bottlenecks and performance of a graph.
int main(int argc, char** argv) {
  absl::SetProgramUsageMessage("Display statistics from MediaPipe log files.");
  absl::ParseCommandLine(argc, argv);

  Reporter reporter;
  reporter.set_compact(absl::GetFlag(FLAGS_compact));
  const auto result = reporter.set_columns(absl::GetFlag(FLAGS_cols));
  if (result.message().length()) {
    std::cout << "WARNING" << std::endl << result.message();
  }

  const auto& flags_logfiles = absl::GetFlag(FLAGS_logfiles);
  for (const auto& file_name : flags_logfiles) {
    std::ifstream ifs(file_name.c_str(), std::ifstream::in);
    mediapipe::proto_ns::io::IstreamInputStream isis(&ifs);
    mediapipe::proto_ns::io::CodedInputStream coded_input_stream(&isis);
    mediapipe::GraphProfile proto;
    if (!proto.ParseFromCodedStream(&coded_input_stream)) {
      std::cerr << "Failed to parse proto.\n";
    } else {
      reporter.Accumulate(proto);
    }
  }
  reporter.Report()->Print(std::cout);
  return 1;
}

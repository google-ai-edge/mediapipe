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

#import <Foundation/Foundation.h>

#include <fstream>
#include <sstream>

#include "absl/strings/match.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/resource_util.h"

namespace mediapipe {

::mediapipe::StatusOr<std::string> PathToResourceAsFile(
    const std::string& path) {
  if (absl::StartsWith(path, "/")) {
    return path;
  }

  NSString* ns_path = [NSString stringWithUTF8String:path.c_str()];
  Class mediapipeGraphClass = NSClassFromString(@"MPPGraph");
  NSString* resource_dir =
      [[NSBundle bundleForClass:mediapipeGraphClass] resourcePath];
  NSString* resolved_ns_path =
      [resource_dir stringByAppendingPathComponent:ns_path];

  std::string resolved_path = [resolved_ns_path UTF8String];
  RET_CHECK([[NSFileManager defaultManager] fileExistsAtPath:resolved_ns_path])
      << "cannot find file: " << resolved_path;

  return resolved_path;
}

::mediapipe::Status GetResourceContents(const std::string& path,
                                        std::string* output) {
  ASSIGN_OR_RETURN(std::string full_path, PathToResourceAsFile(path));

  std::ifstream input_file(full_path);
  std::stringstream buffer;
  buffer << input_file.rdbuf();
  buffer.str().swap(*output);
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe


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

#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/profiler/profiler_resource_util.h"

namespace mediapipe {

StatusOr<std::string> GetDefaultTraceLogDirectory() {
  // Get the Documents directory. iOS apps can write files to this directory.
  NSURL* documents_directory_url = [[[NSFileManager defaultManager]
      URLsForDirectory:NSDocumentDirectory
             inDomains:NSUserDomainMask] lastObject];

  // Note: "createDirectoryAtURL:..." method doesn't successfully create
  // the directory, hence this code uses "createDirectoryAtPath:..".
  NSString* ns_documents_directory = [documents_directory_url path];
  NSError* error;
  BOOL success = [[NSFileManager defaultManager]
            createDirectoryAtPath:ns_documents_directory
      withIntermediateDirectories:YES
                       attributes:nil
                            error:&error];
  if (!success) {
    // TODO: Use NSError+util_status to get status from NSError.
    return absl::InternalError([[error localizedDescription] UTF8String]);
  }

  std::string trace_log_directory = [ns_documents_directory UTF8String];
  return trace_log_directory;
}

}  // namespace mediapipe

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
// Defines DeletingFile.
#include "mediapipe/framework/formats/deleting_file.h"

#include <stdio.h>

#include <utility>

#include "absl/log/absl_log.h"

namespace mediapipe {

DeletingFile::DeletingFile(DeletingFile&& other)
    : path_(std::move(other.path_)),
      delete_on_destruction_(other.delete_on_destruction_) {
  other.delete_on_destruction_ = false;
}

DeletingFile& DeletingFile::operator=(DeletingFile&& other) {
  path_ = std::move(other.path_);
  delete_on_destruction_ = other.delete_on_destruction_;
  other.delete_on_destruction_ = false;
  return *this;
}

DeletingFile::DeletingFile(const std::string& path, bool delete_on_destruction)
    : path_(path), delete_on_destruction_(delete_on_destruction) {}

DeletingFile::~DeletingFile() {
  if (delete_on_destruction_) {
    if (remove(path_.c_str()) != 0) {
      ABSL_LOG(ERROR) << "Unable to delete file: " << path_;
    }
  }
}

const std::string& DeletingFile::Path() const { return path_; }

}  // namespace mediapipe

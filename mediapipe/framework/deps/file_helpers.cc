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

#include "mediapipe/framework/deps/file_helpers.h"

#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include <cerrno>

#include "mediapipe/framework/deps/canonical_errors.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/deps/status_builder.h"

namespace mediapipe {
namespace file {
::mediapipe::Status GetContents(absl::string_view file_name,
                                std::string* output) {
  FILE* fp = fopen(file_name.data(), "r");
  if (fp == NULL) {
    return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "Can't find file: " << file_name;
  }

  output->clear();
  while (!feof(fp)) {
    char buf[4096];
    size_t ret = fread(buf, 1, 4096, fp);
    if (ret == 0 && ferror(fp)) {
      return ::mediapipe::InternalErrorBuilder(MEDIAPIPE_LOC)
             << "Error while reading file: " << file_name;
    }
    output->append(std::string(buf, ret));
  }
  fclose(fp);
  return ::mediapipe::OkStatus();
}

::mediapipe::Status SetContents(absl::string_view file_name,
                                absl::string_view content) {
  FILE* fp = fopen(file_name.data(), "w");
  if (fp == NULL) {
    return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "Can't open file: " << file_name;
  }

  fwrite(content.data(), sizeof(char), content.size(), fp);
  size_t write_error = ferror(fp);
  if (fclose(fp) != 0 || write_error) {
    return ::mediapipe::InternalErrorBuilder(MEDIAPIPE_LOC)
           << "Error while writing file: " << file_name
           << ". Error message: " << strerror(write_error);
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status MatchInTopSubdirectories(
    const std::string& parent_directory, const std::string& file_name,
    std::vector<std::string>* results) {
  DIR* dir = opendir(parent_directory.c_str());
  CHECK(dir);
  // Iterates through the parent direcotry.
  while (true) {
    struct dirent* dir_ent = readdir(dir);
    if (dir_ent == nullptr) {
      break;
    }
    if (std::string(dir_ent->d_name) == "." ||
        std::string(dir_ent->d_name) == "..") {
      continue;
    }
    std::string subpath =
        JoinPath(parent_directory, std::string(dir_ent->d_name));
    DIR* sub_dir = opendir(subpath.c_str());
    // Iterates through the subdirecotry to find file matches.
    while (true) {
      struct dirent* dir_ent_2 = readdir(sub_dir);
      if (dir_ent_2 == nullptr) {
        break;
      }
      if (std::string(dir_ent_2->d_name) == "." ||
          std::string(dir_ent_2->d_name) == "..") {
        continue;
      }
      if (absl::EndsWith(std::string(dir_ent_2->d_name), file_name)) {
        results->push_back(JoinPath(subpath, std::string(dir_ent_2->d_name)));
      }
    }
    closedir(sub_dir);
  }
  closedir(dir);
  return ::mediapipe::OkStatus();
}

::mediapipe::Status MatchFileTypeInDirectory(
    const std::string& directory, const std::string& file_suffix,
    std::vector<std::string>* results) {
  DIR* dir = opendir(directory.c_str());
  CHECK(dir);
  // Iterates through the direcotry.
  while (true) {
    struct dirent* dir_ent = readdir(dir);
    if (dir_ent == nullptr) {
      break;
    }
    if (std::string(dir_ent->d_name) == "." ||
        std::string(dir_ent->d_name) == "..") {
      continue;
    }

    if (absl::EndsWith(std::string(dir_ent->d_name), file_suffix)) {
      results->push_back(JoinPath(directory, std::string(dir_ent->d_name)));
    }
  }
  closedir(dir);
  return ::mediapipe::OkStatus();
}

::mediapipe::Status Exists(absl::string_view file_name) {
  struct stat buffer;
  int status;
  status = stat(file_name.data(), &buffer);
  if (status == 0) {
    return ::mediapipe::OkStatus();
  }
  switch (errno) {
    case EACCES:
      return ::mediapipe::PermissionDeniedError("Insufficient permissions.");
    default:
      return ::mediapipe::NotFoundError("The path does not exist.");
  }
}

}  // namespace file
}  // namespace mediapipe

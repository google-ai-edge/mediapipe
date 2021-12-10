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

#include "mediapipe/framework/deps/file_path.h"

#include "absl/strings/str_cat.h"

namespace mediapipe {
namespace file {

// 40% of the time in JoinPath() is from calls with 2 arguments, so we
// specialize that case.
std::string JoinPath(absl::string_view path1, absl::string_view path2) {
  if (path1.empty()) return std::string(path2);
  if (path2.empty()) return std::string(path1);
  if (path1.back() == '/') {
    if (path2.front() == '/')
      return absl::StrCat(path1, absl::ClippedSubstr(path2, 1));
  } else {
    if (path2.front() != '/') return absl::StrCat(path1, "/", path2);
  }
  return absl::StrCat(path1, path2);
}

namespace internal {

// Given a collection of file paths, append them all together,
// ensuring that the proper path separators are inserted between them.
std::string JoinPathImpl(bool honor_abs,
                         std::initializer_list<absl::string_view> paths) {
  std::string result;

  if (paths.size() != 0) {
    // This size calculation is worst-case: it assumes one extra "/" for every
    // path other than the first.
    size_t total_size = paths.size() - 1;
    for (const absl::string_view& path : paths) total_size += path.size();
    result.resize(total_size);

    auto begin = result.begin();
    auto out = begin;
    bool trailing_slash = false;
    for (absl::string_view path : paths) {
      if (path.empty()) continue;
      if (path.front() == '/') {
        if (honor_abs) {
          out = begin;  // wipe out whatever we've built up so far.
        } else if (trailing_slash) {
          path.remove_prefix(1);
        }
      } else {
        if (!trailing_slash && out != begin) *out++ = '/';
      }
      const size_t this_size = path.size();
      memcpy(&*out, path.data(), this_size);
      out += this_size;
      trailing_slash = out[-1] == '/';
    }
    result.erase(out - begin);
  }
  return result;
}

// Return the parts of the basename of path, split on the final ".".
// If there is no "." in the basename or "." is the final character in the
// basename, the second value will be empty.
std::pair<absl::string_view, absl::string_view> SplitBasename(
    absl::string_view path) {
  path = Basename(path);

  absl::string_view::size_type pos = path.find_last_of('.');
  if (pos == absl::string_view::npos)
    return std::make_pair(path, absl::ClippedSubstr(path, path.size(), 0));
  return std::make_pair(path.substr(0, pos),
                        absl::ClippedSubstr(path, pos + 1));
}

}  // namespace internal

absl::string_view Dirname(absl::string_view path) {
  return SplitPath(path).first;
}

absl::string_view Basename(absl::string_view path) {
  return SplitPath(path).second;
}

std::pair<absl::string_view, absl::string_view> SplitPath(
    absl::string_view path) {
  absl::string_view::size_type pos = path.find_last_of('/');

  // Handle the case with no '/' in 'path'.
  if (pos == absl::string_view::npos)
    return std::make_pair(path.substr(0, 0), path);

  // Handle the case with a single leading '/' in 'path'.
  if (pos == 0)
    return std::make_pair(path.substr(0, 1), absl::ClippedSubstr(path, 1));

  return std::make_pair(path.substr(0, pos),
                        absl::ClippedSubstr(path, pos + 1));
}

absl::string_view Extension(absl::string_view path) {
  return internal::SplitBasename(path).second;
}

}  // namespace file
}  // namespace mediapipe

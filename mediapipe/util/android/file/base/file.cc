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

#include "mediapipe/util/android/file/base/file.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "absl/base/call_once.h"
#include "absl/strings/match.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "mediapipe/framework/port/logging.h"

#ifdef __APPLE__
static_assert(sizeof(off_t) == 8, "Large file support is required");
#define stat64 stat
#define lstat64 lstat
#endif

namespace mediapipe {
namespace file {

bool IsAbsolutePath(const std::string& path) {
  return !path.empty() && (path[0] == '/');
}

Options CreationMode(mode_t permissions) {
  Options options;
  options.set_permissions(permissions);
  return options;
}

}  // namespace file

std::string File::Basename(const std::string& fname) {
  size_t stop = fname.back() == '/' ? fname.size() - 2 : std::string::npos;
  size_t start = fname.find_last_of('/', stop);

  // If no slash, just return.
  if (start == std::string::npos) {
    return fname;
  }

  return fname.substr(start + 1, stop - start);
}

std::string File::StripBasename(const std::string& fname) {
  size_t last_slash = fname.find_last_of('/');

  // If no slash trim all. If already root (or close to it), return that.
  if (last_slash == std::string::npos) {
    return "";
  } else if (fname == "/" || last_slash == 0) {
    return "/";
  }

  return fname.substr(0, last_slash);
}

bool File::IsLocalFile(const std::string& fname) {
  struct stat64 stat_buf;
  int statval = lstat64(fname.c_str(), &stat_buf);
  return statval == 0 && S_ISREG(stat_buf.st_mode);
}

bool File::Exists(const char* name) {
  struct stat64 stat_buf;
  int statval = lstat64(name, &stat_buf);
  return statval == 0;
}

namespace {

absl::once_flag localhost_init;

std::string* localhost_name_str = nullptr;

void LocalHostInit() {
  char buf[256];
  if (gethostname(buf, sizeof(buf)) == 0) {
    buf[sizeof(buf) - 1] = '\0';
    localhost_name_str = new std::string(buf);
  } else {
    LOG(ERROR) << "Could not get local host name";
    localhost_name_str = new std::string("localhost");
  }
}

const std::string* localhost_name() {
  absl::call_once(localhost_init, &LocalHostInit);
  return localhost_name_str;
}

void StringReplace(absl::string_view s, absl::string_view oldsub,
                   absl::string_view newsub, bool replace_all,
                   std::string* res) {
  if (oldsub.empty()) {
    res->append(s.data(), s.length());  // If empty, append the given string.
    return;
  }

  absl::string_view::size_type start_pos = 0;
  absl::string_view::size_type pos;
  do {
    pos = s.find(oldsub, start_pos);
    if (pos == absl::string_view::npos) {
      break;
    }
    res->append(s.data() + start_pos, pos - start_pos);
    res->append(newsub.data(), newsub.length());
    // Start searching again after the "old".
    start_pos = pos + oldsub.length();
  } while (replace_all);
  res->append(s.data() + start_pos, s.length() - start_pos);
}

ptrdiff_t StripDupCharacters(std::string* s, char dup_char,
                             ptrdiff_t start_pos) {
  if (start_pos < 0) start_pos = 0;

  // remove dups by compaction in-place
  ptrdiff_t input_pos = start_pos;   // current reader position
  ptrdiff_t output_pos = start_pos;  // current writer position
  const ptrdiff_t input_end = s->size();
  while (input_pos < input_end) {
    // keep current character
    const char curr_char = (*s)[input_pos];
    if (output_pos != input_pos)  // must copy
      (*s)[output_pos] = curr_char;
    ++input_pos;
    ++output_pos;

    if (curr_char == dup_char) {  // skip subsequent dups
      while ((input_pos < input_end) && ((*s)[input_pos] == dup_char))
        ++input_pos;
    }
  }
  const ptrdiff_t num_deleted = input_pos - output_pos;
  s->resize(s->size() - num_deleted);
  return num_deleted;
}

}  // namespace

static const char* const kSlashDotSlash = "/./";
static const char* const kSlash = "/";

const std::string File::CanonicalizeFileName(const char* fname) {
  const char* nonslash = fname;
  while (*nonslash == '/') ++nonslash;

  // Stubby remote functionality (starts with /remote) removed.

  std::string result;
  absl::string_view fname_piece(fname);
  if (!absl::StrContains(fname_piece, kSlashDotSlash)) {
    result.assign(fname_piece.data(), fname_piece.size());
  } else {
    // repeatedly replace "/./" with "/" until the length stops changing
    std::string fname_string(fname);
    result.reserve(fname_string.size());
    while (true) {
      StringReplace(fname_string, kSlashDotSlash, kSlash, true, &result);
      if (fname_string.size() == result.size()) break;

      // prepare for another go-around
      swap(result, fname_string);
      result.clear();
    }
  }

  // remove repeated '/'s
  StripDupCharacters(&result, '/', 0);
  return result;
}

}  // namespace mediapipe

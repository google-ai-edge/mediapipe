#include "mediapipe/util/str_util.h"

#include <cstdint>

#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"

namespace mediapipe {

namespace {

// Scans for the next newline (\r, \n, or \r\n). It returns the size of the
// newline (1 or 2), and where the newline starts. For example,
//   "hello\r\nworld"  would set 5 as the location and 2 as the newline_size,
//   "\r\r" would set 0 as the location and 1 as the newline_size.
//   "hello world" would return false, and not set location or newline_size.
bool ScanForNewline(const absl::string_view data, int* location,
                    int* newline_size) {
  for (int i = 0; i < data.size(); ++i) {
    if (data[i] == '\r' && i + 1 < data.size() && data[i + 1] == '\n') {
      *location = i;
      *newline_size = 2;
      return true;
    } else if (data[i] == '\r' || data[i] == '\n') {
      *location = i;
      *newline_size = 1;
      return true;
    }
  }
  return false;
}

// Pops the next line ending with "\r", "\n", "\r\n" or <EOF>
bool PopNextLine(const absl::string_view utf8, int* cursor,
                 absl::string_view* line) {
  if (*cursor >= static_cast<int64_t>(utf8.size())) return false;
  int location = 0, newline_size = 0;
  if (ScanForNewline(absl::ClippedSubstr(utf8, *cursor), &location,
                     &newline_size)) {
    *line = absl::string_view(utf8.data() + *cursor, location);
    *cursor += location + newline_size;
  } else {
    *line = absl::string_view(utf8.data() + *cursor, utf8.size() - *cursor);
    *cursor = utf8.size();
  }
  return true;
}

}  // namespace

void ForEachLine(absl::string_view utf8_text,
                 absl::AnyInvocable<void(absl::string_view line)> fn) {
  absl::string_view line;
  int cursor = 0;
  while (PopNextLine(utf8_text, &cursor, &line)) {
    fn(line);
  }
}

}  // namespace mediapipe

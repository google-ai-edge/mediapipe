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

#include "mediapipe/framework/timestamp.h"

#include <string>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"

namespace mediapipe {

// In the following functions:
// - The safe int type will check for overflow/underflow and other errors.
// - The CHECK in the constructor will disallow special values.
TimestampDiff Timestamp::operator-(Timestamp other) const {
  ABSL_CHECK(IsRangeValue() && other.IsRangeValue())
      << "This timestamp is " << DebugString() << " and other was "
      << other.DebugString();
  TimestampBaseType tmp_base = timestamp_ - other.timestamp_;
  return TimestampDiff(tmp_base);
}
TimestampDiff TimestampDiff::operator+(TimestampDiff other) const {
  TimestampBaseType tmp_base = timestamp_ + other.timestamp_;
  return TimestampDiff(tmp_base);
}
TimestampDiff TimestampDiff::operator-(TimestampDiff other) const {
  TimestampBaseType tmp_base = timestamp_ - other.timestamp_;
  return TimestampDiff(tmp_base);
}

// Clamp the addition to the range [Timestamp::Min(), Timestamp::Max()].
Timestamp Timestamp::operator+(TimestampDiff offset) const {
  ABSL_CHECK(IsRangeValue()) << "Timestamp is: " << DebugString();
  TimestampBaseType offset_base(offset.Value());
  if (offset_base >= TimestampBaseType(0)) {
    if (timestamp_.value() >= Timestamp::Max().Value() - offset_base.value()) {
      // We would overflow.
      return Timestamp::Max();
    }
  }
  if (offset_base <= TimestampBaseType(0)) {
    if (timestamp_.value() <= Timestamp::Min().Value() - offset_base.value()) {
      // We would underflow.
      return Timestamp::Min();
    }
  }
  return Timestamp(timestamp_ + offset_base);
}
Timestamp Timestamp::operator-(TimestampDiff offset) const {
  return *this + -offset;
}
Timestamp TimestampDiff::operator+(Timestamp timestamp) const {
  return timestamp + *this;
}

Timestamp Timestamp::operator+=(TimestampDiff other) {
  *this = *this + other;
  return *this;
}
Timestamp Timestamp::operator-=(TimestampDiff other) {
  *this = *this - other;
  return *this;
}
Timestamp Timestamp::operator++() {
  *this += 1;
  return *this;
}
Timestamp Timestamp::operator--() {
  *this -= 1;
  return *this;
}
Timestamp Timestamp::operator++(int /*unused*/) {
  Timestamp previous(*this);
  ++(*this);
  return previous;
}
Timestamp Timestamp::operator--(int /*unused*/) {
  Timestamp previous(*this);
  --(*this);
  return previous;
}

std::string Timestamp::DebugString() const {
  if (IsSpecialValue()) {
    if (*this == Timestamp::Unset()) {
      return "Timestamp::Unset()";
    } else if (*this == Timestamp::Unstarted()) {
      return "Timestamp::Unstarted()";
    } else if (*this == Timestamp::PreStream()) {
      return "Timestamp::PreStream()";
    } else if (*this == Timestamp::Min()) {
      return "Timestamp::Min()";
    } else if (*this == Timestamp::Max()) {
      return "Timestamp::Max()";
    } else if (*this == Timestamp::PostStream()) {
      return "Timestamp::PostStream()";
    } else if (*this == Timestamp::OneOverPostStream()) {
      return "Timestamp::OneOverPostStream()";
    } else if (*this == Timestamp::Done()) {
      return "Timestamp::Done()";
    } else {
      ABSL_LOG(FATAL) << "Unknown special type.";
    }
  }
  return absl::StrCat(timestamp_.value());
}
std::string TimestampDiff::DebugString() const {
  return absl::StrCat(timestamp_.value());
}

Timestamp Timestamp::NextAllowedInStream() const {
  if (*this >= Max() || *this == PreStream()) {
    // Indicates that no further timestamps may occur.
    return OneOverPostStream();
  } else if (*this < Min()) {
    return Min();
  }
  return *this + 1;
}

bool Timestamp::HasNextAllowedInStream() const {
  if (*this >= Max() || *this == PreStream()) {
    return false;
  }
  return true;
}

Timestamp Timestamp::PreviousAllowedInStream() const {
  if (*this <= Min() || *this == PostStream()) {
    // Indicates that no previous timestamps may occur.
    return Unstarted();
  } else if (*this > Max()) {
    return Max();
  }
  return *this - 1;
}

std::ostream& operator<<(std::ostream& os, Timestamp arg) {
  return os << arg.DebugString();
}
std::ostream& operator<<(std::ostream& os, TimestampDiff arg) {
  return os << arg.DebugString();
}

}  // namespace mediapipe

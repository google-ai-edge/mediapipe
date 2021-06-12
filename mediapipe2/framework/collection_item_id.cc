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

#include "mediapipe/framework/collection_item_id.h"

namespace mediapipe {

std::ostream& operator<<(std::ostream& os, CollectionItemId arg) {
  return os << arg.value();
}

CollectionItemId operator+(int lhs, CollectionItemId rhs) { return rhs + lhs; }
CollectionItemId operator-(int lhs, CollectionItemId rhs) { return -rhs + lhs; }
CollectionItemId operator*(int lhs, CollectionItemId rhs) { return rhs * lhs; }

}  // namespace mediapipe

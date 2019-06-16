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

#ifndef MEDIAPIPE_DEPS_STATUS_MATCHERS_H_
#define MEDIAPIPE_DEPS_STATUS_MATCHERS_H_

#include "gtest/gtest.h"
#include "mediapipe/framework/deps/status.h"

// EXPECT_OK marco is already defined in our external dependency library
// protobuf. To be consistent with MEDIAPIPE_EXPECT_OK, we also add prefix
// MEDIAPIPE_ to ASSERT_OK. We prefer to use the marcos with MEDIAPIPE_ prefix
// in mediapipe's codebase.
#define MEDIAPIPE_EXPECT_OK(statement) EXPECT_TRUE((statement).ok())
#define MEDIAPIPE_ASSERT_OK(statement) ASSERT_TRUE((statement).ok())

#endif  // MEDIAPIPE_DEPS_STATUS_MATCHERS_H_

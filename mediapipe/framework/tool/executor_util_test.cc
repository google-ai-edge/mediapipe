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

#include "mediapipe/framework/tool/executor_util.h"

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"

namespace mediapipe {

TEST(GraphTest, MinimumDefaultExecutorStackSizeExistingConfigSizeUnspecified) {
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        executor {
          options {
            [mediapipe.ThreadPoolExecutorOptions.ext] { num_threads: 2 }
          }
        }
      )pb");
  CalculatorGraphConfig expected_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        executor {
          options {
            [mediapipe.ThreadPoolExecutorOptions.ext] {
              num_threads: 2
              stack_size: 131072
            }
          }
        }
      )pb");
  tool::EnsureMinimumDefaultExecutorStackSize(131072, &config);
  EXPECT_THAT(config, EqualsProto(expected_config));
}

TEST(GraphTest, MinimumDefaultExecutorStackSizeExistingConfigSizeTooSmall) {
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        executor {
          options {
            [mediapipe.ThreadPoolExecutorOptions.ext] {
              num_threads: 2
              stack_size: 65536
            }
          }
        }
      )pb");
  CalculatorGraphConfig expected_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        executor {
          options {
            [mediapipe.ThreadPoolExecutorOptions.ext] {
              num_threads: 2
              stack_size: 131072
            }
          }
        }
      )pb");
  tool::EnsureMinimumDefaultExecutorStackSize(131072, &config);
  EXPECT_THAT(config, EqualsProto(expected_config));
}

TEST(GraphTest, MinimumDefaultExecutorStackSizeExistingConfigSizeLargeEnough) {
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        executor {
          options {
            [mediapipe.ThreadPoolExecutorOptions.ext] {
              num_threads: 2
              stack_size: 262144
            }
          }
        }
      )pb");
  CalculatorGraphConfig expected_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        executor {
          options {
            [mediapipe.ThreadPoolExecutorOptions.ext] {
              num_threads: 2
              stack_size: 262144
            }
          }
        }
      )pb");
  tool::EnsureMinimumDefaultExecutorStackSize(131072, &config);
  EXPECT_THAT(config, EqualsProto(expected_config));
}

TEST(GraphTest, MinimumDefaultExecutorStackSizeNumThreads) {
  CalculatorGraphConfig config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        num_threads: 1
      )pb");
  CalculatorGraphConfig expected_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
        executor {
          options {
            [mediapipe.ThreadPoolExecutorOptions.ext] {
              num_threads: 1
              stack_size: 131072
            }
          }
        }
      )pb");
  tool::EnsureMinimumDefaultExecutorStackSize(131072, &config);
  EXPECT_THAT(config, EqualsProto(expected_config));
}

}  // namespace mediapipe

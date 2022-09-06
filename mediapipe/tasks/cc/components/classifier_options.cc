/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/cc/components/classifier_options.h"

#include "mediapipe/tasks/cc/components/classifier_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace components {

tasks::ClassifierOptions ConvertClassifierOptionsToProto(
    ClassifierOptions* options) {
  tasks::ClassifierOptions options_proto;
  options_proto.set_display_names_locale(options->display_names_locale);
  options_proto.set_max_results(options->max_results);
  options_proto.set_score_threshold(options->score_threshold);
  for (const std::string& category : options->category_allowlist) {
    options_proto.add_category_allowlist(category);
  }
  for (const std::string& category : options->category_denylist) {
    options_proto.add_category_denylist(category);
  }
  return options_proto;
}

}  // namespace components
}  // namespace tasks
}  // namespace mediapipe

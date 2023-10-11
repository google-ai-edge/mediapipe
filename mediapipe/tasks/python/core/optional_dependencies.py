# Copyright 2022 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MediaPipe Tasks' common but optional dependencies."""

# TensorFlow isn't a dependency of mediapipe pip package. It's only
# required in the API docgen pipeline so we'll ignore it if tensorflow is not
# installed.
try:
  from tensorflow.tools.docs import doc_controls
except ModuleNotFoundError:
  # Replace the real doc_controls.do_not_generate_docs with an no-op
  doc_controls = lambda: None
  no_op = lambda x: x
  setattr(doc_controls, 'do_not_generate_docs', no_op)

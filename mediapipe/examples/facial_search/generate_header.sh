#!/bin/bash
set -eu
set -o pipefail

[[ $# -ne 1 ]] && echo "Usage: $0  <path/to/images directory>" && exit 1
[[ ! -d "$1" ]] && echo "Not a directory: $1" && exit 2
images=$(python -c "import os; print(os.path.realpath('$1'))")
embeddings=$(python -c "import os; print(os.path.realpath('$(dirname "$0")'))")/embeddings.h
labels=$(python -c "import os; print(os.path.realpath('$(dirname "$0")'))")/labels.h

bazel build \
  --platform_suffix=_cpu \
  -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
  mediapipe/examples/facial_search/desktop:facial_search

cat << EOF >"$embeddings"
// Copyright 2020 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_EXAMPLES_FACIAL_SEARCH_EMBEDDINGS_H_
#define MEDIAPIPE_EXAMPLES_FACIAL_SEARCH_EMBEDDINGS_H_

#include <vector>

namespace mediapipe {

const std::vector<std::vector<float>>& MyEmbeddingsCollection() {
  static const std::vector<std::vector<float>> data = {
EOF

cat << EOF >"$labels"
// Copyright 2020 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_EXAMPLES_FACIAL_SEARCH_LABELS_H_
#define MEDIAPIPE_EXAMPLES_FACIAL_SEARCH_LABELS_H_

#include <vector>

namespace mediapipe {

const std::vector<std::string>& MyCollectionLabels() {
  static const std::vector<std::string> data = {
EOF

pushd "$(dirname "$(dirname "$(dirname "$(dirname "$0")")")")" >/dev/null
while read -r image; do
    case "$image" in
      *BUILD|*.gitignore|*download_images.py) continue
    esac
    echo "Running inference for $(basename "$image")"
    if GLOG_logtostderr=1 ./bazel-bin/mediapipe/examples/facial_search/desktop/facial_search \
        --calculator_graph_config_file=mediapipe/examples/facial_search/graphs/facial_search_cpu.pbtxt \
        --input_video_path="$image" \
        --log_embeddings \
        --without_window \
        2>&1 \
    | grep -Eo '\{.+' >>"$embeddings"; then
        echo '      "'"$(basename "$image")"'",' >>"$labels"
    else
        echo "Failed to process embeddings of $(basename "$image")"
    fi
done < <(find "$images" -type f | sort)
popd

cat << EOF >>"$embeddings"
  };
  return data;
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_EXAMPLES_FACIAL_SEARCH_EMBEDDINGS_H_
EOF

cat << EOF >>"$labels"
  };
  return data;
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_EXAMPLES_FACIAL_SEARCH_LABELS_H_
EOF

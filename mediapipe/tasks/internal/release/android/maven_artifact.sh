#!/bin/bash
# Copyright 2021 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Shell script to package an artifact as a zipped maven repository.

set -o errexit
set -o pipefail

#Parse out the flags
POSITIONAL=()
while [[ $# -gt 0 ]]
do
  flag="$(cut -d'=' -f1 <<< $1)"
  value="$(cut -d'=' -f2 <<< $1)"

  case $flag in
    --group_path)
    FLAGS_group_path="$value"
    ;;

    --artifact_id)
    FLAGS_artifact_id="$value"
    ;;

    --version)
    FLAGS_version="$value"
    ;;

    --artifact)
    FLAGS_artifact="$value"
    ;;

    --source)
    FLAGS_source="$value"
    ;;

    --pom)
    FLAGS_pom="$value"
    ;;

    --metadata)
    FLAGS_metadata="$value"
    ;;

    --output)
    FLAGS_output="$value"
    ;;
  esac

  shift
done
set -- "${POSITIONAL[@]}" # restore positional parameters in case we need that

out_tmp="$(mktemp -d)"
dirname="m2repository"
repo="$out_tmp/$dirname"


mkdir -p "$repo/$FLAGS_group_path/$FLAGS_artifact_id/$FLAGS_version"

cp "$FLAGS_metadata" "$repo/$FLAGS_group_path/$FLAGS_artifact_id/"
cp "$FLAGS_pom" "$repo/$FLAGS_group_path/$FLAGS_artifact_id/$FLAGS_version/"
cp "$FLAGS_artifact" \
    "$repo/$FLAGS_group_path/$FLAGS_artifact_id/$FLAGS_version/"

for file in $(find "$repo" -type f); do
  echo -n "$(sha1sum "$file" | cut -f 1 -d ' ')" > "$file.sha1"
  echo -n "$(md5sum "$file" | cut -f 1 -d ' ')" > "$file.md5"
done

(
  root_dir="$(pwd)"
  cd "$out_tmp"
  # Rare zip options:
  #  -X  no extra file attributes
  #  -0  means no compression
  zip -X -q -r -0 "$root_dir/$FLAGS_output" "$dirname"
)

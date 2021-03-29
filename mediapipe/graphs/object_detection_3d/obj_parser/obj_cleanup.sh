#!/bin/bash

# Copyright 2021 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The SimpleObjParser expects the obj commands to follow v/vt/f order. This
# little script will read all the obj files in a directory and sort the
# existing obj commands inside them to also follow this order (so all v lines
# will appear before all vt lines, which will appear before all f lines).

# Usage: ./obj_cleanup.sh input_folder output_folder
# input_folder and output_folder paths can be absolute or relative.

input_folder=$1
output_folder=$2
if [[ "${input_folder}" == "" ]]; then
  echo "input_folder must be defined.  Usage: ./obj_cleanup.sh input_folder output_folder"
  exit 1
fi
if [[ "${output_folder}" == "" ]]; then
  echo "output_folder must be defined.  Usage: ./obj_cleanup.sh input_folder output_folder"
  exit 1
fi

# Find all the obj files and remove the directory name
# Interestingly, piping | sed 's!.obj!!  also removed the extension obj too.
find "${input_folder}" -name "*.obj" | sed 's!.*/!!' | sort |
while IFS= read -r filename; do
  echo "Clean up ${filename}"
  cat "${input_folder}/${filename}" | grep 'v ' > "${output_folder}/${filename}"
  cat "${input_folder}/${filename}" | grep 'vt ' >> "${output_folder}/${filename}"
  cat "${input_folder}/${filename}" | grep 'f ' >> "${output_folder}/${filename}"
done

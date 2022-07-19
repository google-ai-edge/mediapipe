#!/bin/bash
# Copyright 2022 The Tulsi Authors. All rights reserved.
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
#
# Stub for Xcode's ld invocations to avoid linking but still create the expected
# linker outputs.

set -eu

while test $# -gt 0
do
  case $1 in
  *.dat)
    # Create an empty .dat file containing just a simple header.
    echo -n -e '\x00lld\0' > $1
    ;;
  *)
    ;;
  esac

  shift
done

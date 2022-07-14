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
# Stub for Xcode's clang invocations to avoid compilation but still create the
# expected compiler outputs.

set -eu

while test $# -gt 0
do
  case $1 in
  -MF|--serialize-diagnostics)
    # TODO: See if we can create a valid diagnostics file (it appear to be
    # LLVM bitcode), currently we get warnings like:
    # file.dia:1:1: Could not read serialized diagnostics file: error("Invalid diagnostics signature")
    shift
    touch $1
    ;;
  *.o)
    break
    ;;
  esac

  shift
done

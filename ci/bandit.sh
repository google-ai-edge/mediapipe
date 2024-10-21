#!/bin/bash
# Copyright (c) 2023 Intel Corporation
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
BANDIT_RESULTS="bandit.txt"
function error_handle {
    cat ${BANDIT_RESULTS}
}
function cleanup {
    echo "Cleaning up bandit scan"
    rm ${BANDIT_RESULTS}
}
trap error_handle ERR
trap cleanup EXIT
bandit -r mediapipe/examples/python > ${BANDIT_RESULTS}
if ! grep -FRq "No issues identified." ${BANDIT_RESULTS}; then
    echo "Bandit scan failed for demos";
    exit 1;
fi
rm ${BANDIT_RESULTS}

bandit -r mediapipe/python/solutions/ovms* > ${BANDIT_RESULTS}
if ! grep -FRq "No issues identified." ${BANDIT_RESULTS}; then
    echo "Bandit scan failed for solutions";
    exit 2;
fi

exit 0

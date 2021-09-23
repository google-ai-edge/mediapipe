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
"""MediaPipe Hands connections."""

HAND_PALM_CONNECTIONS = ((0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17))

HAND_THUMB_CONNECTIONS = ((1, 2), (2, 3), (3, 4))

HAND_INDEX_FINGER_CONNECTIONS = ((5, 6), (6, 7), (7, 8))

HAND_MIDDLE_FINGER_CONNECTIONS = ((9, 10), (10, 11), (11, 12))

HAND_RING_FINGER_CONNECTIONS = ((13, 14), (14, 15), (15, 16))

HAND_PINKY_FINGER_CONNECTIONS = ((17, 18), (18, 19), (19, 20))

HAND_CONNECTIONS = frozenset().union(*[
    HAND_PALM_CONNECTIONS, HAND_THUMB_CONNECTIONS,
    HAND_INDEX_FINGER_CONNECTIONS, HAND_MIDDLE_FINGER_CONNECTIONS,
    HAND_RING_FINGER_CONNECTIONS, HAND_PINKY_FINGER_CONNECTIONS
])

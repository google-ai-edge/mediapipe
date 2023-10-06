// Copyright 2023 The MediaPipe Authors.
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

package com.google.mediapipe.tasks.vision.poselandmarker;

import com.google.mediapipe.tasks.components.containers.Connection;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/** Pose landmarks connection constants. */
final class PoseLandmarksConnections {

  @SuppressWarnings("ConstantCaseForConstants")
  static final Set<Connection> POSE_LANDMARKS =
      Collections.unmodifiableSet(
          new HashSet<>(
              Arrays.asList(
                  Connection.create(0, 1),
                  Connection.create(1, 2),
                  Connection.create(2, 3),
                  Connection.create(3, 7),
                  Connection.create(0, 4),
                  Connection.create(4, 5),
                  Connection.create(5, 6),
                  Connection.create(6, 8),
                  Connection.create(9, 10),
                  Connection.create(11, 12),
                  Connection.create(11, 13),
                  Connection.create(13, 15),
                  Connection.create(15, 17),
                  Connection.create(15, 19),
                  Connection.create(15, 21),
                  Connection.create(17, 19),
                  Connection.create(12, 14),
                  Connection.create(14, 16),
                  Connection.create(16, 18),
                  Connection.create(16, 20),
                  Connection.create(16, 22),
                  Connection.create(18, 20),
                  Connection.create(11, 23),
                  Connection.create(12, 24),
                  Connection.create(23, 24),
                  Connection.create(23, 25),
                  Connection.create(24, 26),
                  Connection.create(25, 27),
                  Connection.create(26, 28),
                  Connection.create(27, 29),
                  Connection.create(28, 30),
                  Connection.create(29, 31),
                  Connection.create(30, 32),
                  Connection.create(27, 31),
                  Connection.create(28, 32))));

  private PoseLandmarksConnections() {}
}

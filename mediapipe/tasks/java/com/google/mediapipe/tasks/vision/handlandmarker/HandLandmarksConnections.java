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

package com.google.mediapipe.tasks.vision.handlandmarker;

import com.google.mediapipe.tasks.components.containers.Connection;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/** Hand landmarks connection constants. */
final class HandLandmarksConnections {

  @SuppressWarnings("ConstantCaseForConstants")
  static final Set<Connection> HAND_PALM_CONNECTIONS =
      Collections.unmodifiableSet(
          new HashSet<>(
              Arrays.asList(
                  Connection.create(0, 1),
                  Connection.create(0, 5),
                  Connection.create(9, 13),
                  Connection.create(13, 17),
                  Connection.create(5, 9),
                  Connection.create(0, 17))));

  @SuppressWarnings("ConstantCaseForConstants")
  static final Set<Connection> HAND_THUMB_CONNECTIONS =
      Collections.unmodifiableSet(
          new HashSet<>(
              Arrays.asList(
                  Connection.create(1, 2), Connection.create(2, 3), Connection.create(3, 4))));

  @SuppressWarnings("ConstantCaseForConstants")
  static final Set<Connection> HAND_INDEX_FINGER_CONNECTIONS =
      Collections.unmodifiableSet(
          new HashSet<>(
              Arrays.asList(
                  Connection.create(5, 6), Connection.create(6, 7), Connection.create(7, 8))));

  @SuppressWarnings("ConstantCaseForConstants")
  static final Set<Connection> HAND_MIDDLE_FINGER_CONNECTIONS =
      Collections.unmodifiableSet(
          new HashSet<>(
              Arrays.asList(
                  Connection.create(9, 10), Connection.create(10, 11), Connection.create(11, 12))));

  @SuppressWarnings("ConstantCaseForConstants")
  static final Set<Connection> HAND_RING_FINGER_CONNECTIONS =
      Collections.unmodifiableSet(
          new HashSet<>(
              Arrays.asList(
                  Connection.create(13, 14),
                  Connection.create(14, 15),
                  Connection.create(15, 16))));

  @SuppressWarnings("ConstantCaseForConstants")
  static final Set<Connection> HAND_PINKY_FINGER_CONNECTIONS =
      Collections.unmodifiableSet(
          new HashSet<>(
              Arrays.asList(
                  Connection.create(17, 18),
                  Connection.create(18, 19),
                  Connection.create(19, 20))));

  @SuppressWarnings("ConstantCaseForConstants")
  static final Set<Connection> HAND_CONNECTIONS =
      Collections.unmodifiableSet(
          Stream.of(
                  HAND_PALM_CONNECTIONS.stream(),
                  HAND_THUMB_CONNECTIONS.stream(),
                  HAND_INDEX_FINGER_CONNECTIONS.stream(),
                  HAND_MIDDLE_FINGER_CONNECTIONS.stream(),
                  HAND_RING_FINGER_CONNECTIONS.stream(),
                  HAND_PINKY_FINGER_CONNECTIONS.stream())
              .flatMap(i -> i)
              .collect(Collectors.toSet()));

  private HandLandmarksConnections() {}
}

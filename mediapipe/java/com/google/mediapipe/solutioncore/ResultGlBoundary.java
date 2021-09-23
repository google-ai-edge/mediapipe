// Copyright 2021 The MediaPipe Authors.
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

package com.google.mediapipe.solutioncore;

import com.google.auto.value.AutoValue;

/**
 * The left, right, bottom, and top boundaries of the visible section on the screen. The boundary
 * values are typically within the range -1.0 and 1.0.
 */
@AutoValue
public abstract class ResultGlBoundary {

  static ResultGlBoundary create(float left, float right, float bottom, float top) {
    return new AutoValue_ResultGlBoundary(left, right, bottom, top);
  }

  public abstract float left();

  public abstract float right();

  public abstract float bottom();

  public abstract float top();
}

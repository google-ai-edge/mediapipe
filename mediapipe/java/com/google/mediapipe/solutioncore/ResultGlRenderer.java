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

/** Interface for the customizable MediaPipe solution result OpenGL renderer. */
public interface ResultGlRenderer<T extends ImageSolutionResult> {

  /** Sets up OpenGL rendering when the surface is created or recreated. */
  void setupRendering();

  /**
   * Renders the solution result.
   *
   * @param result a solution result object that contains the solution outputs.
   * @param projectionMatrix a 4 x 4 column-vector matrix stored in column-major order (see also <a
   *     href="https://developer.android.com/reference/android/opengl/Matrix">android.opengl.Matrix</a>).
   *     It is an orthographic projection matrix that maps x and y coordinates in {@code result},
   *     defined in [0, 1]x[0, 1] spanning the entire input image (with a top-left origin), to fit
   *     into the {@link SolutionGlSurfaceView} (with a bottom-left origin) that the input image is
   *     rendered into with potential cropping.
   */
  void renderResult(T result, float[] projectionMatrix);
}

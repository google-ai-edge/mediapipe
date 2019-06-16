// Copyright 2019 The MediaPipe Authors.
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

package com.google.mediapipe.components;

import com.google.mediapipe.framework.TextureFrame;

/** Lightweight abstraction for an object that can receive video frames. */
public interface TextureFrameConsumer {
  /** Called when a new {@link TextureFrame} is available. */
  public abstract void onNewFrame(TextureFrame frame);
}

/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.google.mediapipe.tasks.core;

/** Represents a text part of a multi-modal content block. */
public final class TextPart extends Part {
  private final String text;

  /**
   * Creates a new {@link TextPart} with the specified text.
   *
   * @param text the text content.
   */
  public TextPart(String text) {
    this.text = text;
  }

  /**
   * Returns the text content of this part.
   *
   * @return the text content.
   */
  public String getText() {
    return text;
  }
}

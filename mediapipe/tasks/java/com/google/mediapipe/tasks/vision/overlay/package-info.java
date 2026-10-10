// Copyright 2026 The MediaPipe Authors.
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

/**
 * Overlay helpers so Jetpack Compose (and XML) apps can draw MediaPipe Tasks vision results on
 * top of a camera preview or still image.
 *
 * <p>The Tasks AAR cannot depend on Compose without forcing Compose onto every Android client.
 * These classes use {@link android.graphics.Canvas}, which Compose already interops with:
 *
 * <ul>
 *   <li>{@link com.google.mediapipe.tasks.vision.overlay.VisionOverlayView} inside {@code
 *       AndroidView} — same pattern as CameraX {@code PreviewView}.
 *   <li>{@link com.google.mediapipe.tasks.vision.overlay.VisionOverlayRenderer} from a Compose
 *       {@code Canvas} via {@code drawIntoCanvas}.
 * </ul>
 *
 * <p>Scale matches the official samples: IMAGE/VIDEO use FIT_START ({@code min} scale), LIVE_STREAM
 * uses FILL_START ({@code max} scale) so landmarks line up with CameraX {@code PreviewView} in
 * {@code FILL_START} / {@code FILL_CENTER} for typical 4:3 models.
 */
package com.google.mediapipe.tasks.vision.overlay;

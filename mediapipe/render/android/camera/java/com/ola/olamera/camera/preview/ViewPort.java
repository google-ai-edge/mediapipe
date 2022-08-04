/*
 * Copyright 2020 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.ola.olamera.camera.preview;

import android.view.Surface;
import android.view.SurfaceView;

import androidx.annotation.IntDef;
import androidx.annotation.RestrictTo;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;


public final class ViewPort {

    /**
     * LayoutDirection that defines the start and end of the {@link ScaleType}.
     *
     * @hide
     * @see android.util.LayoutDirection
     */
    @RestrictTo(RestrictTo.Scope.LIBRARY_GROUP)
    @IntDef({android.util.LayoutDirection.LTR, android.util.LayoutDirection.RTL})
    @Retention(RetentionPolicy.SOURCE)
    public @interface LayoutDirection {
    }

    /**
     * Scale types used to calculate the crop rect for a {@link UseCase}.
     *
     * @hide
     */
    @RestrictTo(RestrictTo.Scope.LIBRARY_GROUP)
    @IntDef({FILL_START, FILL_CENTER, FILL_END, FIT, CENTER_INSIDE})
    @Retention(RetentionPolicy.SOURCE)
    public @interface ScaleType {
    }

    /**
     * Generate a crop rect that once applied, it scales the output while maintaining its aspect
     * ratio, so it fills the entire {@link ViewPort}, and align it to the start of the
     * {@link ViewPort}, which is the top left corner in a left-to-right (LTR) layout, or the top
     * right corner in a right-to-left (RTL) layout.
     * <p>
     * This may cause the output to be cropped if the output aspect ratio does not match that of
     * the {@link ViewPort}.
     */
    public static final int FILL_START = 0;

    /**
     * Generate a crop rect that once applied, it scales the output while maintaining its aspect
     * ratio, so it fills the entire {@link ViewPort} and center it.
     * <p>
     * This may cause the output to be cropped if the output aspect ratio does not match that of
     * the {@link ViewPort}.
     */
    public static final int FILL_CENTER = 1;

    /**
     * Generate a crop rect that once applied, it scales the output while maintaining its aspect
     * ratio, so it fills the entire {@link ViewPort}, and align it to the end of the
     * {@link ViewPort}, which is the bottom right corner in a left-to-right (LTR) layout, or the
     * bottom left corner in a right-to-left (RTL) layout.
     * <p>
     * This may cause the output to be cropped if the output aspect ratio does not match that of
     * the {@link ViewPort}.
     */
    public static final int FILL_END = 2;

    /**
     * Generate the max possible crop rect ignoring the aspect ratio. For {@link ImageAnalysis}
     * and {@link ImageCapture}, the output will be an image defined by the crop rect.
     *
     * <p> For {@link Preview}, further calculation is needed to to fit the crop rect into the
     * viewfinder. Code sample below is a simplified version assuming {@link Surface}
     * orientation is the same as the camera sensor orientation, the viewfinder is a
     * {@link SurfaceView} and the viewfinder's pixel width/height is the same as the size
     * request by CameraX in {@link SurfaceRequest#getResolution()}. For more complicated
     * scenarios, please check out the source code of PreviewView in androidx.camera.view artifact.
     *
     * <p> First, calculate the transformation to fit the crop rect in the center of the viewfinder:
     *
     * <pre>{@code
     *   val transformation = Matrix()
     *   transformation.setRectToRect(
     *       cropRect, new RectF(0, 0, viewFinder.width, viewFinder.height, ScaleToFit.CENTER))
     * }</pre>
     *
     * <p> Then apply the transformation to the viewfinder:
     *
     * <pre>{@code
     *   val transformedRect = RectF(0, 0, viewFinder.width, viewFinder.height)
     *   transformation.mapRect(surfaceRect)
     *   viewFinder.pivotX = 0
     *   viewFinder.pivotY = 0
     *   viewFinder.translationX = transformedRect.left
     *   viewFinder.translationY = transformedRect.top
     *   viewFinder.scaleX = surfaceRect.width/transformedRect.width
     *   viewFinder.scaleY = surfaceRect.height/transformedRect.height
     * }</pre>
     */
    public static final int FIT = 3;


    public static final int CENTER_INSIDE = 4;


}

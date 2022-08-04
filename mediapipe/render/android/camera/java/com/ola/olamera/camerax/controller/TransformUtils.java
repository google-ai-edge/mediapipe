/*
 * Copyright 2021 The Android Open Source Project
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

package com.ola.olamera.camerax.controller;

import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.RectF;
import android.media.ExifInterface;
import android.util.Size;
import android.view.Surface;

import androidx.annotation.NonNull;
import androidx.annotation.RestrictTo;

/**
 * Utility class for transform.
 *
 * <p> The vertices representation uses a float array to represent a rectangle with arbitrary
 * rotation and rotation-direction. It could be otherwise represented by a triple of a
 * {@link RectF}, a rotation degrees integer and a boolean flag for the rotation-direction
 * (clockwise v.s. counter-clockwise).
 *
 * TODO(b/179827713): merge this with {@link androidx.camera.core.internal.utils.ImageUtil}.
 *
 * @hide
 */
@RestrictTo(RestrictTo.Scope.LIBRARY_GROUP)
public class TransformUtils {

    // Normalized space (-1, -1) - (1, 1).
    public static final RectF NORMALIZED_RECT = new RectF(-1, -1, 1, 1);

    private TransformUtils() {
    }

    /**
     * Gets the size of the {@link Rect}.
     */
    @NonNull
    public static Size rectToSize(@NonNull Rect rect) {
        return new Size(rect.width(), rect.height());
    }

    /**
     * Converts an array of vertices to a {@link RectF}.
     */
    @NonNull
    public static RectF verticesToRect(@NonNull float[] vertices) {
        return new RectF(
                min(vertices[0], vertices[2], vertices[4], vertices[6]),
                min(vertices[1], vertices[3], vertices[5], vertices[7]),
                max(vertices[0], vertices[2], vertices[4], vertices[6]),
                max(vertices[1], vertices[3], vertices[5], vertices[7])
        );
    }

    /**
     * Returns the max value.
     */
    public static float max(float value1, float value2, float value3, float value4) {
        return Math.max(Math.max(value1, value2), Math.max(value3, value4));
    }

    /**
     * Returns the min value.
     */
    public static float min(float value1, float value2, float value3, float value4) {
        return Math.min(Math.min(value1, value2), Math.min(value3, value4));
    }

    /**
     * Converts {@link Surface} rotation to rotation degrees: 90, 180, 270 or 0.
     */
    public static int surfaceRotationToRotationDegrees(int rotationValue) {
        switch (rotationValue) {
            case Surface.ROTATION_0:
                return 0;
            case Surface.ROTATION_90:
                return 90;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_270:
                return 270;
            default:
                throw new IllegalStateException("Unexpected rotation value " + rotationValue);
        }
    }

    /**
     * Returns true if the rotation degrees is 90 or 270.
     */
    public static boolean is90or270(int rotationDegrees) {
        if (rotationDegrees == 90 || rotationDegrees == 270) {
            return true;
        }
        if (rotationDegrees == 0 || rotationDegrees == 180) {
            return false;
        }
        throw new IllegalArgumentException("Invalid rotation degrees: " + rotationDegrees);
    }

    /**
     * Converts a {@link Size} to a float array of vertices.
     */
    @NonNull
    public static float[] sizeToVertices(@NonNull Size size) {
        return new float[]{0, 0, size.getWidth(), 0, size.getWidth(), size.getHeight(), 0,
                size.getHeight()};
    }

    /**
     * Converts a {@link RectF} defined by top, left, right and bottom to an array of vertices.
     */
    @NonNull
    public static float[] rectToVertices(@NonNull RectF rectF) {
        return new float[]{rectF.left, rectF.top, rectF.right, rectF.top, rectF.right, rectF.bottom,
                rectF.left, rectF.bottom};
    }

    /**
     * Checks if aspect ratio matches while tolerating rounding error.
     *
     * <p> One example of the usage is comparing the viewport-based crop rect from different use
     * cases. The crop rect is rounded because pixels are integers, which may introduce an error
     * when we check if the aspect ratio matches. For example, when PreviewView's
     * width/height are prime numbers 601x797, the crop rect from other use cases cannot have a
     * matching aspect ratio even if they are based on the same viewport. This method checks the
     * aspect ratio while tolerating a rounding error.
     *
     * @param size1       the rounded size1
     * @param isAccurate1 if size1 is accurate. e.g. it's true if it's the PreviewView's
     *                    dimension which viewport is based on
     * @param size2       the rounded size2
     * @param isAccurate2 if size2 is accurate.
     */
    public static boolean isAspectRatioMatchingWithRoundingError(
            @NonNull Size size1, boolean isAccurate1, @NonNull Size size2, boolean isAccurate2) {
        // The crop rect coordinates are rounded values. Each value is at most .5 away from their
        // true values. So the width/height, which is the difference of 2 coordinates, are at most
        // 1.0 away from their true value.
        // First figure out the possible range of the aspect ratio's ture value.
        float ratio1UpperBound;
        float ratio1LowerBound;
        if (isAccurate1) {
            ratio1UpperBound = (float) size1.getWidth() / size1.getHeight();
            ratio1LowerBound = ratio1UpperBound;
        } else {
            ratio1UpperBound = (size1.getWidth() + 1F) / (size1.getHeight() - 1F);
            ratio1LowerBound = (size1.getWidth() - 1F) / (size1.getHeight() + 1F);
        }
        float ratio2UpperBound;
        float ratio2LowerBound;
        if (isAccurate2) {
            ratio2UpperBound = (float) size2.getWidth() / size2.getHeight();
            ratio2LowerBound = ratio2UpperBound;
        } else {
            ratio2UpperBound = (size2.getWidth() + 1F) / (size2.getHeight() - 1F);
            ratio2LowerBound = (size2.getWidth() - 1F) / (size2.getHeight() + 1F);
        }
        // Then we check if the true value range overlaps.
        return ratio1UpperBound >= ratio2LowerBound && ratio2UpperBound >= ratio1LowerBound;
    }

    /**
     * Gets the transform from one {@link Rect} to another with rotation degrees.
     *
     * <p> Following is how the source is mapped to the target with a 90° rotation. The rect
     * <a, b, c, d> is mapped to <a', b', c', d'>.
     *
     * <pre>
     *  a----------b               d'-----------a'
     *  |  source  |    -90°->     |            |
     *  d----------c               |   target   |
     *                             |            |
     *                             c'-----------b'
     * </pre>
     */
    @NonNull
    public static Matrix getRectToRect(
            @NonNull RectF source, @NonNull RectF target, int rotationDegrees) {
        // Map source to normalized space.
        Matrix matrix = new Matrix();
        matrix.setRectToRect(source, NORMALIZED_RECT, Matrix.ScaleToFit.FILL);
        // Add rotation.
        matrix.postRotate(rotationDegrees);
        // Restore the normalized space to target's coordinates.
        matrix.postConcat(getNormalizedToBuffer(target));
        return matrix;
    }

    /**
     * Gets the transform from a normalized space (-1, -1) - (1, 1) to the given rect.
     */
    @NonNull
    public static Matrix getNormalizedToBuffer(@NonNull Rect viewPortRect) {
        return getNormalizedToBuffer(new RectF(viewPortRect));
    }

    /**
     * Gets the transform from a normalized space (-1, -1) - (1, 1) to the given rect.
     */
    @NonNull
    private static Matrix getNormalizedToBuffer(@NonNull RectF viewPortRect) {
        Matrix normalizedToBuffer = new Matrix();
        normalizedToBuffer.setRectToRect(NORMALIZED_RECT, viewPortRect, Matrix.ScaleToFit.FILL);
        return normalizedToBuffer;
    }

    /**
     * Gets the transform matrix based on exif orientation.
     */
    @NonNull
    public static Matrix getExifTransform(int exifOrientation, int width, int height) {
        Matrix matrix = new Matrix();

        // Map the bitmap to a normalized space and perform transform. It's more readable, and it
        // can be tested with Robolectric's ShadowMatrix (Matrix#setPolyToPoly is currently not
        // shadowed by ShadowMatrix).
        RectF rect = new RectF(0, 0, width, height);
        matrix.setRectToRect(rect, NORMALIZED_RECT, Matrix.ScaleToFit.FILL);

        // A flag that checks if the image has been rotated 90/270.
        boolean isWidthHeightSwapped = false;

        // Transform the normalized space based on exif orientation.
        switch (exifOrientation) {
            case ExifInterface.ORIENTATION_FLIP_HORIZONTAL:
                matrix.postScale(-1f, 1f);
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                matrix.postRotate(180);
                break;
            case ExifInterface.ORIENTATION_FLIP_VERTICAL:
                matrix.postScale(1f, -1f);
                break;
            case ExifInterface.ORIENTATION_TRANSPOSE:
                // Flipped about top-left <--> bottom-right axis, it can also be represented by
                // flip horizontally and then rotate 270 degree clockwise.
                matrix.postScale(-1f, 1f);
                matrix.postRotate(270);
                isWidthHeightSwapped = true;
                break;
            case ExifInterface.ORIENTATION_ROTATE_90:
                matrix.postRotate(90);
                isWidthHeightSwapped = true;
                break;
            case ExifInterface.ORIENTATION_TRANSVERSE:
                // Flipped about top-right <--> bottom left axis, it can also be represented by
                // flip horizontally and then rotate 90 degree clockwise.
                matrix.postScale(-1f, 1f);
                matrix.postRotate(90);
                isWidthHeightSwapped = true;
                break;
            case ExifInterface.ORIENTATION_ROTATE_270:
                matrix.postRotate(270);
                isWidthHeightSwapped = true;
                break;
            case ExifInterface.ORIENTATION_NORMAL:
                // Fall-through
            case ExifInterface.ORIENTATION_UNDEFINED:
                // Fall-through
            default:
                break;
        }

        // Map the normalized space back to the bitmap coordinates.
        RectF restoredRect = isWidthHeightSwapped ? new RectF(0, 0, height, width) : rect;
        Matrix restore = new Matrix();
        restore.setRectToRect(NORMALIZED_RECT, restoredRect, Matrix.ScaleToFit.FILL);
        matrix.postConcat(restore);

        return matrix;
    }
}

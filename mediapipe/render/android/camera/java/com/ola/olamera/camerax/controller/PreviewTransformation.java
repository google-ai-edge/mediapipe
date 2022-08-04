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

package com.ola.olamera.camerax.controller;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.util.LayoutDirection;
import android.util.Size;
import android.view.Display;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.TextureView;
import android.view.View;

import com.ola.olamera.camerax.controller.internal.compat.quirk.PreviewOneThirdWiderQuirk;
import com.ola.olamera.camerax.controller.internal.compat.quirk.TextureViewRotationQuirk;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.VisibleForTesting;
import androidx.camera.camera2.internal.compat.quirk.DeviceQuirks;
import androidx.camera.core.Logger;
import androidx.camera.core.SurfaceRequest;
import androidx.core.util.Preconditions;

import static android.graphics.Paint.ANTI_ALIAS_FLAG;
import static android.graphics.Paint.DITHER_FLAG;
import static android.graphics.Paint.FILTER_BITMAP_FLAG;
/**
 * Handles {PreviewView} transformation.
 *
 * <p> This class transforms the camera output and display it in a PreviewView. The goal is
 * to transform it in a way so that the entire area of
 * {@link SurfaceRequest.TransformationInfo#getCropRect()} is 1) visible to end users, and 2)
 * displayed as large as possible.
 *
 * <p> The inputs for the calculation are 1) the dimension of the Surface, 2) the crop rect, 3) the
 * dimension of the PreviewView and 4) rotation degrees:
 *
 * <pre>
 * Source: +-----Surface-----+     Destination:  +-----PreviewView----+
 *         |                 |                   |                    |
 *         |  +-crop rect-+  |                   |                    |
 *         |  |           |  |                   +--------------------+
 *         |  |           |  |
 *         |  |    -->    |  |        Rotation:        <-----+
 *         |  |           |  |                           270°|
 *         |  |           |  |                               |
 *         |  +-----------+  |
 *         +-----------------+
 *
 * By mapping the Surface crop rect to match the PreviewView, we have:
 *
 *  +------transformed Surface-------+
 *  |                                |
 *  |     +----PreviewView-----+     |
 *  |     |          ^         |     |
 *  |     |          |         |     |
 *  |     +--------------------+     |
 *  |                                |
 *  +--------------------------------+
 * </pre>
 *
 * <p> The transformed Surface is how the PreviewView's inner view should behave, to make the
 * crop rect matches the PreviewView.
 */
final class PreviewTransformation {

    private static final String TAG = "PreviewTransform";


    // SurfaceRequest.getResolution().
    private Size mResolution;
    // This represents the area of the Surface that should be visible to end users. The value
    // is based on TransformationInfo.getCropRect() with possible corrections due to device quirks.
    private Rect mSurfaceCropRect;
    // This rect represents the size of the viewport in preview. It's always the same as
    // TransformationInfo.getCropRect().
    private Rect mViewportRect;
    // TransformationInfo.getRotationDegrees().
    private int mPreviewRotationDegrees;
    // TransformationInfo.getTargetRotation.
    private int mTargetRotation;
    // Whether the preview is using front camera.
    private boolean mIsFrontCamera;


    PreviewTransformation() {
    }

    /**
     * Sets the inputs.
     *
     * <p> All the values originally come from a {@link SurfaceRequest}.
     */
    @SuppressLint({"RestrictedApi", "UnsafeExperimentalUsageError"})
    void setTransformationInfo(@NonNull SurfaceRequest.TransformationInfo transformationInfo,
            Size resolution, boolean isFrontCamera) {
        Logger.d(TAG, "Transformation info set: " + transformationInfo + " " + resolution + " "
                + isFrontCamera);
        mSurfaceCropRect = getCorrectedCropRect(transformationInfo.getCropRect());
        mViewportRect = transformationInfo.getCropRect();
        mPreviewRotationDegrees = transformationInfo.getRotationDegrees();
        mTargetRotation = transformationInfo.getTargetRotation();
        mResolution = resolution;
        mIsFrontCamera = isFrontCamera;
    }

    /**
     * Creates a matrix that makes {@link TextureView}'s rotation matches the
     * {@link #mTargetRotation}.
     *
     * <p> The value should be applied by calling {@link TextureView#setTransform(Matrix)}. Usually
     * {@link #mTargetRotation} is the display rotation. In that case, this
     * matrix will just make a {@link TextureView} works like a {@link SurfaceView}. If not, then
     * it will further correct it to the desired rotation.
     *
     * <p> This method is also needed in {@link #createTransformedBitmap} to correct the screenshot.
     */
    @SuppressLint("RestrictedApi")
    @VisibleForTesting
    Matrix getTextureViewCorrectionMatrix() {
        Preconditions.checkState(isTransformationInfoReady());
        RectF surfaceRect = new RectF(0, 0, mResolution.getWidth(), mResolution.getHeight());
        @SuppressLint("RestrictedApi") int rotationDegrees = -TransformUtils.surfaceRotationToRotationDegrees(mTargetRotation);

        TextureViewRotationQuirk textureViewRotationQuirk =
                DeviceQuirks.get(TextureViewRotationQuirk.class);
        if (textureViewRotationQuirk != null) {
            rotationDegrees += textureViewRotationQuirk.getCorrectionRotation(mIsFrontCamera);
        }
        return TransformUtils.getRectToRect(surfaceRect, surfaceRect, rotationDegrees);
    }

    /**
     * Calculates the transformation and applies it to the inner view ofPreviewView.
     *
     * <p> The inner view could be {@link SurfaceView} or a {@link TextureView}.
     * {@link TextureView} needs a preliminary correction since it doesn't handle the
     * display rotation.
     */
    @SuppressLint("RestrictedApi")
    void transformView(Size previewViewSize, int layoutDirection, @NonNull View preview) {
        if (previewViewSize.getHeight() == 0 || previewViewSize.getWidth() == 0) {
            Logger.w(TAG, "Transform not applied due to PreviewView size: " + previewViewSize);
            return;
        }
        if (!isTransformationInfoReady()) {
            return;
        }

        if (preview instanceof TextureView) {
            // For TextureView, correct the orientation to match the target rotation.
            ((TextureView) preview).setTransform(getTextureViewCorrectionMatrix());
        } else {
            // Logs an error if non-display rotation is used with SurfaceView.
            Display display = preview.getDisplay();
            if (display != null && display.getRotation() != mTargetRotation) {
                Logger.e(TAG, "Non-display rotation not supported with SurfaceView / PERFORMANCE "
                        + "mode.");
            }
        }

        RectF surfaceRectInPreviewView = getTransformedSurfaceRect(previewViewSize,
                layoutDirection);
        preview.setPivotX(0);
        preview.setPivotY(0);
        preview.setScaleX(surfaceRectInPreviewView.width() / mResolution.getWidth());
        preview.setScaleY(surfaceRectInPreviewView.height() / mResolution.getHeight());
        preview.setTranslationX(surfaceRectInPreviewView.left - preview.getLeft());
        preview.setTranslationY(surfaceRectInPreviewView.top - preview.getTop());
    }

    /**
     * Gets the transformed {@link Surface} rect in PreviewView coordinates.
     *
     * <p> Returns desired rect of the inner view that once applied, the only part visible to
     * end users is the crop rect.
     */
    @SuppressLint("RestrictedApi")
    private RectF getTransformedSurfaceRect(Size previewViewSize, int layoutDirection) {
        Preconditions.checkState(isTransformationInfoReady());
        Matrix surfaceToPreviewView =
                getSurfaceToPreviewViewMatrix(previewViewSize, layoutDirection);
        RectF rect = new RectF(0, 0, mResolution.getWidth(), mResolution.getHeight());
        surfaceToPreviewView.mapRect(rect);
        return rect;
    }

    /**
     * Calculates the transformation from {@link Surface} coordinates to PreviewView
     * coordinates.
     *
     * <p> The calculation is based on making the crop rect to fill or fit the PreviewView.
     */
    @SuppressLint("RestrictedApi")
    Matrix getSurfaceToPreviewViewMatrix(Size previewViewSize, int layoutDirection) {
        Preconditions.checkState(isTransformationInfoReady());

        // Get the target of the mapping, the coordinates of the crop rect in PreviewView.
        RectF previewViewCropRect;
        if (isViewportAspectRatioMatchPreviewView(previewViewSize)) {
            // If crop rect has the same aspect ratio as PreviewView, scale the crop rect to fill
            // the entire PreviewView. This happens if the scale type is FILL_* AND a
            // PreviewView-based viewport is used.
            previewViewCropRect = new RectF(0, 0, previewViewSize.getWidth(),
                    previewViewSize.getHeight());
        } else {
            // If the aspect ratios don't match, it could be 1) scale type is FIT_*, 2) the
            // Viewport is not based on the PreviewView or 3) both.
            previewViewCropRect = getPreviewViewViewportRectForMismatchedAspectRatios(
                    previewViewSize, layoutDirection);
        }
        Matrix matrix = TransformUtils.getRectToRect(new RectF(mSurfaceCropRect), previewViewCropRect,
                mPreviewRotationDegrees);
        if (mIsFrontCamera) {
            // SurfaceView/TextureView automatically mirrors the Surface for front camera, which
            // needs to be compensated by mirroring the Surface around the upright direction of the
            // output image.
            if (TransformUtils.is90or270(mPreviewRotationDegrees)) {
                // If the rotation is 90/270, the Surface should be flipped vertically.
                //   +---+     90 +---+  270 +---+
                //   | ^ | -->    | < |      | > |
                //   +---+        +---+      +---+
                matrix.preScale(1F, -1F, mSurfaceCropRect.centerX(), mSurfaceCropRect.centerY());
            } else {
                // If the rotation is 0/180, the Surface should be flipped horizontally.
                //   +---+      0 +---+  180 +---+
                //   | ^ | -->    | ^ |      | v |
                //   +---+        +---+      +---+
                matrix.preScale(-1F, 1F, mSurfaceCropRect.centerX(), mSurfaceCropRect.centerY());
            }
        }
        return matrix;
    }

    /**
     * Gets the vertices of the crop rect in Surface.
     */
    @SuppressLint("RestrictedApi")
    private Rect getCorrectedCropRect(Rect surfaceCropRect) {
        PreviewOneThirdWiderQuirk quirk = DeviceQuirks.get(PreviewOneThirdWiderQuirk.class);
        if (quirk != null) {
            // Correct crop rect if the device has a quirk.
            RectF cropRectF = new RectF(surfaceCropRect);
            Matrix correction = new Matrix();
            correction.setScale(
                    quirk.getCropRectScaleX(),
                    1f,
                    surfaceCropRect.centerX(),
                    surfaceCropRect.centerY());
            correction.mapRect(cropRectF);
            Rect correctRect = new Rect();
            cropRectF.round(correctRect);
            return correctRect;
        }
        return surfaceCropRect;
    }

    /**
     * Gets the viewport rect in PreviewView coordinates for the case where viewport's
     * aspect ratio doesn't match PreviewView's aspect ratio.
     *
     * <p> When aspect ratios don't match, additional calculation is needed to figure out how to
     * fit crop rect into the PreviewView
     */
    RectF getPreviewViewViewportRectForMismatchedAspectRatios(Size previewViewSize,
            int layoutDirection) {
        RectF previewViewRect = new RectF(0, 0, previewViewSize.getWidth(),
                previewViewSize.getHeight());
        Size rotatedViewportSize = getRotatedViewportSize();
        RectF rotatedViewportRect = new RectF(0, 0, rotatedViewportSize.getWidth(),
                rotatedViewportSize.getHeight());
        Matrix matrix = new Matrix();
        setMatrixRectToRect(matrix, rotatedViewportRect, previewViewRect);
        matrix.mapRect(rotatedViewportRect);
        if (layoutDirection == LayoutDirection.RTL) {
            return flipHorizontally(rotatedViewportRect, (float) previewViewSize.getWidth() / 2);
        }
        return rotatedViewportRect;
    }

    /**
     * Set the matrix that maps the source rectangle to the destination rectangle.
     *
     * <p> This static method is an extension of {@link Matrix#setRectToRect} with an additional
     * support for FILL_* types.
     */
    private static void setMatrixRectToRect(Matrix matrix, RectF source, RectF destination) {
        Matrix.ScaleToFit matrixScaleType = Matrix.ScaleToFit.FILL;
        // TODO: 后续可能需要更改ScaleType
//        boolean isFitTypes =
//                scaleType == FIT_CENTER || scaleType == FIT_START || scaleType == FIT_END;
        boolean isFitTypes = false;
        if (isFitTypes) {
            matrix.setRectToRect(source, destination, matrixScaleType);
        } else {
            // android.graphics.Matrix doesn't support fill scale types. The workaround is
            // mapping inversely from destination to source, then invert the matrix.
            matrix.setRectToRect(destination, source, matrixScaleType);
            matrix.invert(matrix);
        }
    }

    /**
     * Flips the given rect along a vertical line for RTL layout direction.
     */
    private static RectF flipHorizontally(RectF original, float flipLineX) {
        return new RectF(
                flipLineX + flipLineX - original.right,
                original.top,
                flipLineX + flipLineX - original.left,
                original.bottom);
    }

    /**
     * Returns viewport size with target rotation applied.
     */
    @SuppressLint("RestrictedApi")
    private Size getRotatedViewportSize() {
        if (TransformUtils.is90or270(mPreviewRotationDegrees)) {
            return new Size(mViewportRect.height(), mViewportRect.width());
        }
        return new Size(mViewportRect.width(), mViewportRect.height());
    }

    /**
     * Checks if the viewport's aspect ratio matches that of the PreviewView.
     */
    @SuppressLint("RestrictedApi")
    @VisibleForTesting
    boolean isViewportAspectRatioMatchPreviewView(Size previewViewSize) {
        // Using viewport rect to check if the viewport is based on the PreviewView.
        Size rotatedViewportSize = getRotatedViewportSize();
        return TransformUtils.isAspectRatioMatchingWithRoundingError(
                previewViewSize, /* isAccurate1= */ true,
                rotatedViewportSize,  /* isAccurate2= */ false);
    }

    /**
     * Return the crop rect of the preview surface.
     */
    @Nullable
    Rect getSurfaceCropRect() {
        return mSurfaceCropRect;
    }

    /**
     * Creates a transformed screenshot of PreviewView.
     *
     * <p> Creates the transformed {@link Bitmap} by applying the same transformation applied to
     * the inner view. T
     *
     * @param original a snapshot of the untransformed inner view.
     */
    Bitmap createTransformedBitmap(@NonNull Bitmap original, Size previewViewSize,
            int layoutDirection) {
        if (!isTransformationInfoReady()) {
            return original;
        }
        Matrix textureViewCorrection = getTextureViewCorrectionMatrix();
        RectF surfaceRectInPreviewView = getTransformedSurfaceRect(previewViewSize,
                layoutDirection);

        Bitmap transformed = Bitmap.createBitmap(
                previewViewSize.getWidth(), previewViewSize.getHeight(), original.getConfig());
        Canvas canvas = new Canvas(transformed);

        Matrix canvasTransform = new Matrix();
        canvasTransform.postConcat(textureViewCorrection);
        canvasTransform.postScale(surfaceRectInPreviewView.width() / mResolution.getWidth(),
                surfaceRectInPreviewView.height() / mResolution.getHeight());
        canvasTransform.postTranslate(surfaceRectInPreviewView.left, surfaceRectInPreviewView.top);

        canvas.drawBitmap(original, canvasTransform,
                new Paint(ANTI_ALIAS_FLAG | FILTER_BITMAP_FLAG | DITHER_FLAG));
        return transformed;
    }

    /**
     * Calculates the mapping from a UI touch point (0, 0) - (width, height) to normalized
     * space (-1, -1) - (1, 1).
     *
     * <p> This is used by {@link PreviewViewMeteringPointFactory}.
     *
     * @return null if transformation info is not set.
     */
    @Nullable
    Matrix getPreviewViewToNormalizedSurfaceMatrix(Size previewViewSize, int layoutDirection) {
        if (!isTransformationInfoReady()) {
            return null;
        }
        Matrix matrix = new Matrix();

        // Map PreviewView coordinates to Surface coordinates.
        getSurfaceToPreviewViewMatrix(previewViewSize, layoutDirection).invert(matrix);

        // Map Surface coordinates to normalized coordinates (-1, -1) - (1, 1).
        Matrix normalization = new Matrix();
        normalization.setRectToRect(
                new RectF(0, 0, mResolution.getWidth(), mResolution.getHeight()),
                new RectF(0, 0, 1, 1), Matrix.ScaleToFit.FILL);
        matrix.postConcat(normalization);

        return matrix;
    }

    private boolean isTransformationInfoReady() {
        return mSurfaceCropRect != null && mResolution != null;
    }
}

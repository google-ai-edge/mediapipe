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

import android.annotation.SuppressLint;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.RectF;
import android.os.Build;
import android.util.LayoutDirection;
import android.util.Rational;
import android.util.Size;

import androidx.annotation.IntRange;
import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

import com.ola.olamera.util.CameraLogger;
import com.ola.olamera.util.CameraShould;
import com.ola.olamera.util.ImageUtils;

/**
 * Utility methods for calculating viewports.
 */
public class ViewPorts {
    private ViewPorts() {

    }

    /**
     * Calculate a set of ViewPorts based on the combination of the camera, viewport, and use cases.
     *
     * <p> This method calculates the crop rect for each use cases. It only thinks in abstract terms
     * like the original dimension, output rotation and desired crop rect expressed via viewport.
     * It does not care about the use case types or the device/display rotation.
     *
     * @param fullSensorRect        The full size of the viewport.
     * @param viewPortAspectRatio   The aspect ratio of the viewport.
     * @param outputRotationDegrees Clockwise rotation to correct the surfaces to display
     *                              rotation.
     * @param scaleType             The scale type to calculate
     * @param layoutDirection       The direction of layout.
     * @param surfaceOutResolution  The resolutions of the UseCases
     * @return The set of Viewports that should be set for each UseCase
     */
    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    @NonNull
    public static Rect calculateViewPortRect(
            @NonNull Rect fullSensorRect,
            boolean isFrontCamera,
            @NonNull Rational viewPortAspectRatio,
            @IntRange(from = 0, to = 359) int outputRotationDegrees,
            @ViewPort.ScaleType int scaleType,
            @ViewPort.LayoutDirection int layoutDirection,
            @NonNull Size surfaceOutResolution) {
        CameraShould.beTrue(
                fullSensorRect.width() > 0 && fullSensorRect.height() > 0,
                "Cannot compute viewport crop rects zero sized sensor rect.");

        CameraLogger.i("ViewPorts", "calculateViewPortRect { \n\t\t fullSensorRect:%s \n\t\t isFrontCamera:%b " +
                        "\n\t\t viewPortAspectRatio:%s \n\t\t outputRotationDegrees:%d \n\t\t scaleType:%d \n\t\t " +
                        "surfaceOutResolution:%s \n}", fullSensorRect, isFrontCamera, viewPortAspectRatio,
                outputRotationDegrees, scaleType, surfaceOutResolution);

        // The key to calculate the crop rect is that all the crop rect should match to the same
        // region on camera sensor. This method first calculates the shared camera region, and then
        // maps it use cases to find out their crop rects.

        // Calculate the mapping between sensor buffer and UseCases, and the sensor rect shared
        // by all use cases.
        RectF fullSensorRectF = new RectF(fullSensorRect);
        RectF sensorIntersectionRect = new RectF(fullSensorRect);
        // Calculate the transformation from UseCase to sensor.
        Matrix useCaseToSensorTransformation = new Matrix();
        RectF srcRect = new RectF(0, 0, surfaceOutResolution.getWidth(),
                surfaceOutResolution.getHeight());
        useCaseToSensorTransformation.setRectToRect(srcRect, fullSensorRectF,
                Matrix.ScaleToFit.CENTER);
//        useCaseToSensorTransformations.put(entry.getKey(), useCaseToSensorTransformation);

        // Calculate the UseCase intersection in sensor coordinates.
        RectF useCaseSensorRect = new RectF();
        useCaseToSensorTransformation.mapRect(useCaseSensorRect, srcRect);
        sensorIntersectionRect.intersect(useCaseSensorRect);

        // Crop the shared sensor rect based on viewport parameters.
        Rational rotatedViewPortAspectRatio = ImageUtils.getRotatedAspectRatio(
                outputRotationDegrees, viewPortAspectRatio);
        RectF viewPortRect = getScaledRect(
                sensorIntersectionRect, rotatedViewPortAspectRatio, scaleType, isFrontCamera,
                layoutDirection, outputRotationDegrees);

        // Map the cropped shared sensor rect to UseCase coordinates.
        RectF useCaseOutputRect = new RectF();
        Matrix sensorToUseCaseTransformation = new Matrix();
        // Transform the sensor crop rect to UseCase coordinates.
        useCaseToSensorTransformation.invert(sensorToUseCaseTransformation);
        sensorToUseCaseTransformation.mapRect(useCaseOutputRect, viewPortRect);
        Rect outputCropRect = new Rect();
        useCaseOutputRect.round(outputCropRect);
        return outputCropRect;
    }

    /**
     * Returns the container rect that the given rect fills.
     *
     * <p> For FILL types, returns the largest container rect that is smaller than the view port.
     * The returned rectangle is also required to 1) have the view port's aspect ratio and 2) be
     * in the surface coordinates.
     *
     * <p> For FIT, returns the largest possible rect shared by all use cases.
     */
    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    @SuppressLint("SwitchIntDef")
    @NonNull
    public static RectF getScaledRect(
            @NonNull RectF fittingRect,
            @NonNull Rational containerAspectRatio,
            @ViewPort.ScaleType int scaleType,
            boolean isFrontCamera,
            @ViewPort.LayoutDirection int layoutDirection,
            @IntRange(from = 0, to = 359) int rotationDegrees) {
        if (scaleType == ViewPort.FIT) {
            // Return the fitting rect if the rect is fully covered by the container.
            return fittingRect;
        }
        // Using Matrix' convenience methods fill the rect into the containing rect with given
        // aspect ratio.
        // NOTE: By using the Matrix#setRectToRect, we assume the "start" is always (0, 0) and
        // the "end" is always (w, h), which is NOT always true depending on rotation, layout
        // orientation and/or camera lens facing. We need to correct the rect based on rotation and
        // layout direction.
        Matrix viewPortToSurfaceTransformation = new Matrix();
        RectF viewPortRect = new RectF(0, 0, containerAspectRatio.getNumerator(),
                containerAspectRatio.getDenominator());
        switch (scaleType) {
            case ViewPort.FILL_CENTER:
                viewPortToSurfaceTransformation.setRectToRect(
                        viewPortRect, fittingRect, Matrix.ScaleToFit.CENTER);
                break;
            case ViewPort.FILL_START:
                viewPortToSurfaceTransformation.setRectToRect(
                        viewPortRect, fittingRect, Matrix.ScaleToFit.START);
                break;
            case ViewPort.FILL_END:
                viewPortToSurfaceTransformation.setRectToRect(
                        viewPortRect, fittingRect, Matrix.ScaleToFit.END);
                break;
            case ViewPort.CENTER_INSIDE:
                //not crop
                viewPortToSurfaceTransformation.setRectToRect(
                        fittingRect, fittingRect, Matrix.ScaleToFit.FILL);
                break;
            default:
                throw new IllegalStateException("Unexpected scale type: " + scaleType);
        }

        RectF viewPortRectInSurfaceCoordinates = new RectF();
        viewPortToSurfaceTransformation.mapRect(viewPortRectInSurfaceCoordinates, viewPortRect);

        // Correct the crop rect based on rotation and layout direction.
        return correctStartOrEnd(
                shouldMirrorStartAndEnd(isFrontCamera, layoutDirection),
                rotationDegrees,
                fittingRect,
                viewPortRectInSurfaceCoordinates);
    }

    /**
     * Correct viewport based on rotation and layout direction.
     *
     * <p> Both rotation and mirroring change the definition of the "start" and "end" in
     * scale type. For rotation, since the value is clockwise rotation should be applied to the
     * output buffer, the start/end point should be rotated counterclockwisely. If mirroring is
     * needed, the start/end point should be mirrored based on the upright direction of the
     * image.
     */
    private static RectF correctStartOrEnd(boolean isMirrored,
                                           @IntRange(from = 0, to = 359) int rotationDegrees,
                                           RectF containerRect,
                                           RectF cropRect) {
        // For each scenario there is an illustration of the output buffer without correction.
        // The arrow represents the opposite direction of gravity. The start/end point should
        // rotate counterclockwisely based on rotationDegrees, and mirror along the line of the
        // arrow if mirroring is needed.

        //
        // Start +-----+
        //       |  ^  |
        //       +-----+  End
        //
        boolean ltrRotation0 = rotationDegrees == 0 && !isMirrored;
        //
        // Start +-----+     90°     +-----+ End  Mirrored  Start +-----+
        //       |  ^  |    ===>     |  <  |        ==>           |  <  |
        //       +-----+ End   Start +-----+                      +-----+ End
        //
        boolean rtlRotation90 = rotationDegrees == 90 && isMirrored;
        if (ltrRotation0 || rtlRotation90) {
            return cropRect;
        }

        //
        // Start +-----+ Mirrored  +-----+ Start
        //       |  ^  |   ===>    |  ^  |
        //       +-----+ End   End +-----+
        //
        boolean rtlRotation0 = rotationDegrees == 0 && isMirrored;
        //
        // Start +-----+   270°   +-----+ Start
        //       |  ^  |   ===>   |  >  |
        //       +-----+ End  End +-----+
        //
        boolean ltrRotation270 = rotationDegrees == 270 && !isMirrored;
        if (rtlRotation0 || ltrRotation270) {
            return flipHorizontally(cropRect, containerRect.centerX());
        }

        //
        // Start +-----+    90°     +-----+ End
        //       |  ^  |   ===>     |  <  |
        //       +-----+ End  Start +-----+
        //
        boolean ltrRotation90 = rotationDegrees == 90 && !isMirrored;
        //
        // Start +-----+   180°  End +-----+   Mirrored    +-----+ End
        //       |  ^  |   ===>      |  v  |     ==>       |  v  |
        //       +-----+ End         +-----+ Start   Start +-----+
        //
        boolean rtlRotation180 = rotationDegrees == 180 && isMirrored;
        if (ltrRotation90 || rtlRotation180) {
            return flipVertically(cropRect, containerRect.centerY());
        }

        //
        // Start +-----+   180°  End +-----+
        //       |  ^  |   ===>      |  v  |
        //       +-----+ End         +-----+ Start
        //
        boolean ltrRotation180 = rotationDegrees == 180 && !isMirrored;
        //
        // Start +-----+   270°   +-----+ Start  Mirrored  End +-----+
        //       |  ^  |   ===>   |  >  |           ==>        |  >  |
        //       +-----+ End  End +-----+                      +-----+ Start
        //
        boolean rtlRotation270 = rotationDegrees == 270 && isMirrored;
        if (ltrRotation180 || rtlRotation270) {
            return flipHorizontally(flipVertically(cropRect, containerRect.centerY()),
                    containerRect.centerX());
        }

        throw new IllegalArgumentException("Invalid argument: mirrored " + isMirrored + " "
                + "rotation " + rotationDegrees);
    }

    /**
     * Checks if the start/end direction in scale type should be mirrored.
     *
     * <p> They should be mirrored if one and only one of the following is true: the front camera is
     * used or layout direction is RTL.
     */
    private static boolean shouldMirrorStartAndEnd(boolean isFrontCamera,
                                                   @ViewPort.LayoutDirection int layoutDirection) {
        return isFrontCamera ^ layoutDirection == LayoutDirection.RTL;
    }

    private static RectF flipHorizontally(RectF original, float flipLineX) {
        return new RectF(
                flipX(original.right, flipLineX),
                original.top,
                flipX(original.left, flipLineX),
                original.bottom);
    }

    private static RectF flipVertically(RectF original, float flipLineY) {
        return new RectF(
                original.left,
                flipY(original.bottom, flipLineY),
                original.right,
                flipY(original.top, flipLineY));
    }

    private static float flipX(float x, float flipLineX) {
        return flipLineX + flipLineX - x;
    }

    private static float flipY(float y, float flipLineY) {
        return flipLineY + flipLineY - y;
    }
}

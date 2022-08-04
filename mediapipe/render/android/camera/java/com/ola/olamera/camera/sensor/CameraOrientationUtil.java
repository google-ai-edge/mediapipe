/*
 * Copyright (C) 2019 The Android Open Source Project
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

package com.ola.olamera.camera.sensor;

import android.util.Log;
import android.view.Surface;

import androidx.annotation.RestrictTo;
import androidx.annotation.RestrictTo.Scope;

import com.ola.olamera.camera.session.CameraSelector;

/**
 * Contains utility methods related to camera orientation.
 *
 * @hide
 */
@RestrictTo(Scope.LIBRARY_GROUP)
public final class CameraOrientationUtil {
    private static final String TAG = "CameraOrientationUtil";
    private static final boolean DEBUG = false;

    // Do not allow instantiation
    private CameraOrientationUtil() {
    }

    /**
     * Calculates the delta between a source rotation and destination rotation.
     *
     * <p>A typical use of this method would be calculating the angular difference between the
     * display orientation (destRotationDegrees) and camera sensor orientation
     * (sourceRotationDegrees).
     *
     * @param destRotationDegrees   The destination rotation relative to the device's natural
     *                              rotation.
     * @param sourceRotationDegrees The source rotation relative to the device's natural rotation.
     * @param isOppositeFacing      Whether the source and destination planes are facing opposite
     *                              directions.
     */
    public static int getRelativeImageRotation(
            int destRotationDegrees, int sourceRotationDegrees, boolean isOppositeFacing) {
        int result;
        if (isOppositeFacing) {
            result = (sourceRotationDegrees - destRotationDegrees + 360) % 360;
        } else {
            result = (sourceRotationDegrees + destRotationDegrees) % 360;
        }
        if (DEBUG) {
            Log.d(
                    TAG,
                    String.format(
                            "getRelativeImageRotation: destRotationDegrees=%s, "
                                    + "sourceRotationDegrees=%s, isOppositeFacing=%s, "
                                    + "result=%s",
                            destRotationDegrees, sourceRotationDegrees, isOppositeFacing, result));
        }
        return result;
    }

    /**
     * Converts rotation values enumerated in {@link Surface} to their equivalent in degrees.
     *
     * <p>Valid values for the relative rotation are {@link Surface#ROTATION_0}, {@link
     * Surface#ROTATION_90}, {@link Surface#ROTATION_180}, {@link Surface#ROTATION_270}.
     *
     * @param rotationEnum One of the enumerated rotation values from {@link Surface}.
     * @return The equivalent rotation value in degrees.
     * @throws IllegalArgumentException If the provided rotation enum is not one of those defined in
     *                                  {@link Surface}.
     */
    public static int surfaceRotationToDegrees(int rotationEnum) {
        int rotationDegrees;
        switch (rotationEnum) {
            case Surface.ROTATION_0:
                rotationDegrees = 0;
                break;
            case Surface.ROTATION_90:
                rotationDegrees = 90;
                break;
            case Surface.ROTATION_180:
                rotationDegrees = 180;
                break;
            case Surface.ROTATION_270:
                rotationDegrees = 270;
                break;
            default:
                throw new IllegalArgumentException("Unsupported surface rotation: " + rotationEnum);
        }

        return rotationDegrees;
    }

    /**
     * 用户视觉上图片旋转角度
     */
    public static int getCameraImageRotation(CameraSelector.CameraLenFacing lenFacing, int cameraSensorRotation, int displaySensorRotation) {
        int rotation = lenFacing == CameraSelector.CameraLenFacing.LEN_FACING_BACK ? (360 - displaySensorRotation)
                : displaySensorRotation;
        return (cameraSensorRotation + rotation) % 360;
    }
}

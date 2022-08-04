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

import android.content.Context;
import android.view.OrientationEventListener;
import android.view.Surface;

import androidx.annotation.NonNull;


public abstract class RotationReceiver {

    private static final int INVALID_SURFACE_ROTATION = -1;

    // Synthetic access
    @SuppressWarnings("WeakerAccess")
    int mRotation = INVALID_SURFACE_ROTATION;

    private final OrientationEventListener mOrientationEventListener;

    public RotationReceiver(@NonNull Context context) {
        mOrientationEventListener = new OrientationEventListener(context) {
            @Override
            public void onOrientationChanged(int orientation) {
                if (orientation == OrientationEventListener.ORIENTATION_UNKNOWN) {
                    // Short-circuit if orientation is unknown. Unknown rotation can't be handled
                    // so it shouldn't be sent.
                    return;
                }

                int newRotation;
                if (orientation >= 315 || orientation < 45) {
                    newRotation = Surface.ROTATION_0;
                } else if (orientation >= 225) {
                    newRotation = Surface.ROTATION_90;
                } else if (orientation >= 135) {
                    newRotation = Surface.ROTATION_180;
                } else {
                    newRotation = Surface.ROTATION_270;
                }
                if (mRotation != newRotation) {
                    mRotation = newRotation;
                    onRotationChanged(newRotation);
                }
            }
        };
    }

    /**
     * Checks if the RotationReceiver can detect orientation changes.
     *
     * @see OrientationEventListener#canDetectOrientation()
     */
    public boolean canDetectOrientation() {
        return mOrientationEventListener.canDetectOrientation();
    }

    /**
     * Enables the RotationReceiver so it will monitor the sensor and call onRotationChanged when
     * the device orientation changes.
     *
     * <p> By default, the receiver is not enabled.
     *
     * @see OrientationEventListener#enable()
     */
    public void enable() {
        mOrientationEventListener.enable();
    }

    /**
     * Disables the RotationReceiver.
     *
     * @see OrientationEventListener#disable()
     */
    public void disable() {
        mOrientationEventListener.disable();
    }

    /**
     * Called when the physical rotation of the device changes.
     *
     * <p> The rotation is one of the {@link Surface} rotations mapped from orientation
     * degrees.
     *
     * <table summary="Orientation degrees to Surface rotation mapping">
     * <tr><th>Orientation degrees</th><th>Surface rotation</th></tr>
     * <tr><td>[-45°, 45°)</td><td>{@link Surface#ROTATION_0}</td></tr>
     * <tr><td>[45°, 135°)</td><td>{@link Surface#ROTATION_270}</td></tr>
     * <tr><td>[135°, 225°)</td><td>{@link Surface#ROTATION_180}</td></tr>
     * <tr><td>[225°, 315°)</td><td>{@link Surface#ROTATION_90}</td></tr>
     * </table>
     *
     * @see OrientationEventListener#onOrientationChanged(int)
     */
    public abstract void onRotationChanged(int rotation);
}

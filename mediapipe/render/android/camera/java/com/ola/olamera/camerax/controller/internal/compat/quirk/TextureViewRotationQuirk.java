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

package com.ola.olamera.camerax.controller.internal.compat.quirk;

import android.os.Build;
import android.view.TextureView;

import androidx.camera.core.impl.Quirk;

/**
 * A quirk that requires applying extra rotation on {@link TextureView}
 *
 * <p> On certain devices, the rotation of the output is incorrect. One example is b/177561470.
 * In which case, the extra rotation is needed to correct the output on {@link TextureView}.
 */
public class TextureViewRotationQuirk implements Quirk {

    private static final String FAIRPHONE = "Fairphone";
    private static final String FAIRPHONE_2_MODEL = "FP2";

    static boolean load() {
        return isFairphone2();
    }

    /**
     * Gets correction needed for the given camera.
     */
    public int getCorrectionRotation(boolean isFrontCamera) {
        if (isFairphone2() && isFrontCamera) {
            // On Fairphone2, the front camera output on TextureView is rotated 180°.
            // See: b/177561470.
            return 180;
        }
        return 0;
    }

    private static boolean isFairphone2() {
        return FAIRPHONE.equalsIgnoreCase(Build.MANUFACTURER)
                && FAIRPHONE_2_MODEL.equalsIgnoreCase(Build.MODEL);
    }
}

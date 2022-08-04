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

import androidx.camera.core.impl.Quirk;

/**
 * A quirk where the preview buffer is stretched.
 *
 * <p> The symptom is, the preview's FOV is always 1/3 wider than intended. For example, if the
 * preview Surface is 800x600, it's actually has a FOV of 1066x600 with the same center point,
 * but squeezed to fit the 800x600 buffer.
 */
public class PreviewOneThirdWiderQuirk implements Quirk {

    private static final String SAMSUNG_A3_2017 = "A3Y17LTE"; // b/180121821
    private static final String SAMSUNG_J5_PRIME = "ON5XELTE"; // b/183329599

    static boolean load() {
        boolean isSamsungJ5PrimeAndApi26 =
                SAMSUNG_J5_PRIME.equals(Build.DEVICE.toUpperCase()) && Build.VERSION.SDK_INT >= 26;
        boolean isSamsungA3 = SAMSUNG_A3_2017.equals(Build.DEVICE.toUpperCase());
        return isSamsungJ5PrimeAndApi26 || isSamsungA3;
    }

    /**
     * The mount that the crop rect needs to be scaled in x.
     */
    public float getCropRectScaleX() {
        return 0.75f;
    }
}

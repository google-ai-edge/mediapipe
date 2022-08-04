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
 * A quirk where SurfaceView is stretched.
 *
 * <p> On Samsung Galaxy Z Fold2, transform APIs (e.g. View#setScaleX) do not work as intended.
 * b/129403806
 */
public class SurfaceViewStretchedQuirk implements Quirk {

    // Samsung Galaxy Z Fold2 b/129403806
    private static final String SAMSUNG = "SAMSUNG";
    private static final String GALAXY_Z_FOLD_2 = "F2Q";

    static boolean load() {
        return SAMSUNG.equals(Build.MANUFACTURER.toUpperCase()) && GALAXY_Z_FOLD_2.equals(
                Build.DEVICE.toUpperCase());
    }
}

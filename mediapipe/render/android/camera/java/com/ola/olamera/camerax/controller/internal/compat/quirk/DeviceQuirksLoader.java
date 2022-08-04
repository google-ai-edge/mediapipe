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

import java.util.ArrayList;
import java.util.List;

import androidx.annotation.NonNull;
import androidx.camera.core.impl.Quirk;

/**
 * Loads all device specific quirks required for the current device
 */
public class DeviceQuirksLoader {

    private DeviceQuirksLoader() {
    }

    /**
     * Goes through all defined device-specific quirks, and returns those that should be loaded
     * on the current device.
     */
    @NonNull
    static List<Quirk> loadQuirks() {
        final List<Quirk> quirks = new ArrayList<>();

        // Load all device specific quirks
        if (PreviewOneThirdWiderQuirk.load()) {
            quirks.add(new PreviewOneThirdWiderQuirk());
        }

        if (SurfaceViewStretchedQuirk.load()) {
            quirks.add(new SurfaceViewStretchedQuirk());
        }

        if (TextureViewRotationQuirk.load()) {
            quirks.add(new TextureViewRotationQuirk());
        }

        return quirks;
    }
}

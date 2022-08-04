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

import android.annotation.SuppressLint;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.impl.Quirk;
import androidx.camera.core.impl.Quirks;

/**
 * Provider of device specific quirks for the view module, which are used for device specific
 * workarounds.
 * <p>
 * Device specific quirks depend on device properties, including the manufacturer
 * ({@link android.os.Build#MANUFACTURER}), model ({@link android.os.Build#MODEL}) and OS
 * level ({@link android.os.Build.VERSION#SDK_INT}).
 * <p>
 * Device specific quirks are lazily loaded, i.e. They are loaded the first time they're needed.
 */
@SuppressLint("RestrictedApi")
public class DeviceQuirks {

    @NonNull
    private static final Quirks QUIRKS;

    static {
        QUIRKS = new Quirks(DeviceQuirksLoader.loadQuirks());
    }

    private DeviceQuirks() {
    }

    /**
     * Retrieves a specific device {@link Quirk} instance given its type.
     *
     * @param quirkClass The type of device quirk to retrieve.
     * @return A device {@link Quirk} instance of the provided type, or {@code null} if it isn't
     * found.
     */
    @Nullable
    public static <T extends Quirk> T get(@NonNull final Class<T> quirkClass) {
        return QUIRKS.get(quirkClass);
    }
}

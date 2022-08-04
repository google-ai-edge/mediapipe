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

package com.ola.olamera.camera.session;

import androidx.annotation.RestrictTo;
import androidx.annotation.RestrictTo.Scope;

/**
 * A report of failed capture for a single image capture.
 *
 * @hide
 */
@RestrictTo(Scope.LIBRARY_GROUP)
public final class CameraCaptureFailure {

    private final Reason mReason;

    /** @hide */
    @RestrictTo(Scope.LIBRARY_GROUP)
    public CameraCaptureFailure(Reason reason) {
        mReason = reason;
    }

    /**
     * Determine why the request was dropped, whether due to an error or to a user action.
     *
     * @return int The reason code.
     * @see Reason#ERROR
     */
    public Reason getReason() {
        return mReason;
    }

    /**
     * The capture result has been dropped this frame only due to an error in the framework.
     *
     * @see #getReason()
     */
    public enum Reason {
        ERROR,
    }
}

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

import android.hardware.camera2.CaptureResult;

import androidx.annotation.NonNull;
import androidx.annotation.RestrictTo;
import androidx.annotation.RestrictTo.Scope;


/**
 * The result of a single image capture.
 *
 * @hide
 */
@RestrictTo(Scope.LIBRARY_GROUP)
public interface CameraCaptureResult {

    /**
     * Returns the current auto focus mode of operation.
     */
    @NonNull
    CameraCaptureMetaData.AfMode getAfMode();

    /**
     * Returns the current auto focus state.
     */
    @NonNull
    CameraCaptureMetaData.AfState getAfState();

    /**
     * Returns the current auto exposure state.
     */
    @NonNull
    CameraCaptureMetaData.AeState getAeState();

    /**
     * Returns the current auto white balance state.
     */
    @NonNull
    CameraCaptureMetaData.AwbState getAwbState();

    /**
     * Returns the current flash state.
     */
    @NonNull
    CameraCaptureMetaData.FlashState getFlashState();

    /**
     * Returns the timestamp in nanoseconds.
     *
     * <p> If the timestamp was unavailable then it will return {@code -1L}.
     */
    long getTimestamp();

    /**
     * Returns the tag associated with the capture request.
     */
    Object getTag();

    CaptureResult getCaptureResult();


    /**
     * An implementation of CameraCaptureResult which always return default results.
     */
    final class EmptyCameraCaptureResult implements CameraCaptureResult {

        public static CameraCaptureResult create() {
            return new EmptyCameraCaptureResult();
        }

        @NonNull
        @Override
        public CameraCaptureMetaData.AfMode getAfMode() {
            return CameraCaptureMetaData.AfMode.UNKNOWN;
        }

        @NonNull
        @Override
        public CameraCaptureMetaData.AfState getAfState() {
            return CameraCaptureMetaData.AfState.UNKNOWN;
        }

        @NonNull
        @Override
        public CameraCaptureMetaData.AeState getAeState() {
            return CameraCaptureMetaData.AeState.UNKNOWN;
        }

        @NonNull
        @Override
        public CameraCaptureMetaData.AwbState getAwbState() {
            return CameraCaptureMetaData.AwbState.UNKNOWN;
        }

        @NonNull
        @Override
        public CameraCaptureMetaData.FlashState getFlashState() {
            return CameraCaptureMetaData.FlashState.UNKNOWN;
        }

        @Override
        public long getTimestamp() {
            return -1L;
        }

        @Override
        public Object getTag() {
            return null;
        }

        @Override
        public CaptureResult getCaptureResult() {
            return null;
        }
    }
}

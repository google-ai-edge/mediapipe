/*
 * Copyright 2019 The Android Open Source Project
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
import android.os.Build;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;

import com.ola.olamera.camera.session.CameraCaptureMetaData.AeState;
import com.ola.olamera.camera.session.CameraCaptureMetaData.AfMode;
import com.ola.olamera.camera.session.CameraCaptureMetaData.AfState;
import com.ola.olamera.camera.session.CameraCaptureMetaData.AwbState;
import com.ola.olamera.camera.session.CameraCaptureMetaData.FlashState;

/**
 * The camera2 implementation for the capture result of a single image capture.
 */
@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
final class Camera2CameraCaptureResult implements CameraCaptureResult {
    private static final String TAG = "C2CameraCaptureResult";

    private final Object mTag;

    /**
     * The actual camera2 {@link CaptureResult}.
     */
    private final CaptureResult mCaptureResult;

    Camera2CameraCaptureResult(@Nullable Object tag, CaptureResult captureResult) {
        mTag = tag;
        mCaptureResult = captureResult;
    }

    /**
     * Converts the camera2 {@link CaptureResult#CONTROL_AF_MODE} to {@link AfMode}.
     *
     * @return the {@link AfMode}.
     */
    @NonNull
    @Override
    public AfMode getAfMode() {
        Integer mode = mCaptureResult.get(CaptureResult.CONTROL_AF_MODE);
        if (mode == null) {
            return AfMode.UNKNOWN;
        }
        switch (mode) {
            case CaptureResult.CONTROL_AF_MODE_OFF:
            case CaptureResult.CONTROL_AF_MODE_EDOF:
                return AfMode.OFF;
            case CaptureResult.CONTROL_AF_MODE_AUTO:
            case CaptureResult.CONTROL_AF_MODE_MACRO:
                return AfMode.ON_MANUAL_AUTO;
            case CaptureResult.CONTROL_AF_MODE_CONTINUOUS_PICTURE:
            case CaptureResult.CONTROL_AF_MODE_CONTINUOUS_VIDEO:
                return AfMode.ON_CONTINUOUS_AUTO;
            default: // fall out
        }
        Log.e(TAG, "Undefined af mode: " + mode);
        return AfMode.UNKNOWN;
    }

    /**
     * Converts the camera2 {@link CaptureResult#CONTROL_AF_STATE} to {@link AfState}.
     *
     * @return the {@link AfState}.
     */
    @NonNull
    @Override
    public AfState getAfState() {
        Integer state = mCaptureResult.get(CaptureResult.CONTROL_AF_STATE);
        if (state == null) {
            return AfState.UNKNOWN;
        }
        switch (state) {
            case CaptureResult.CONTROL_AF_STATE_INACTIVE:
                return AfState.INACTIVE;
            case CaptureResult.CONTROL_AF_STATE_ACTIVE_SCAN:
            case CaptureResult.CONTROL_AF_STATE_PASSIVE_SCAN:
            case CaptureResult.CONTROL_AF_STATE_PASSIVE_UNFOCUSED:
                return AfState.SCANNING;
            case CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED:
                return AfState.LOCKED_FOCUSED;
            case CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED:
                return AfState.LOCKED_NOT_FOCUSED;
            case CaptureResult.CONTROL_AF_STATE_PASSIVE_FOCUSED:
                return AfState.FOCUSED;
            default: // fall out
        }
        Log.e(TAG, "Undefined af state: " + state);
        return AfState.UNKNOWN;
    }

    /**
     * Converts the camera2 {@link CaptureResult#CONTROL_AE_STATE} to {@link AeState}.
     *
     * @return the {@link AeState}.
     */
    @NonNull
    @Override
    public AeState getAeState() {
        Integer state = mCaptureResult.get(CaptureResult.CONTROL_AE_STATE);
        if (state == null) {
            return AeState.UNKNOWN;
        }
        switch (state) {
            case CaptureResult.CONTROL_AE_STATE_INACTIVE:
                return AeState.INACTIVE;
            case CaptureResult.CONTROL_AE_STATE_SEARCHING:
            case CaptureResult.CONTROL_AE_STATE_PRECAPTURE:
                return AeState.SEARCHING;
            case CaptureResult.CONTROL_AE_STATE_FLASH_REQUIRED:
                return AeState.FLASH_REQUIRED;
            case CaptureResult.CONTROL_AE_STATE_CONVERGED:
                return AeState.CONVERGED;
            case CaptureResult.CONTROL_AE_STATE_LOCKED:
                return AeState.LOCKED;
            default: // fall out
        }
        Log.e(TAG, "Undefined ae state: " + state);
        return AeState.UNKNOWN;
    }

    /**
     * Converts the camera2 {@link CaptureResult#CONTROL_AWB_STATE} to {@link AwbState}.
     *
     * @return the {@link AwbState}.
     */
    @NonNull
    @Override
    public AwbState getAwbState() {
        Integer state = mCaptureResult.get(CaptureResult.CONTROL_AWB_STATE);
        if (state == null) {
            return AwbState.UNKNOWN;
        }
        switch (state) {
            case CaptureResult.CONTROL_AWB_STATE_INACTIVE:
                return AwbState.INACTIVE;
            case CaptureResult.CONTROL_AWB_STATE_SEARCHING:
                return AwbState.METERING;
            case CaptureResult.CONTROL_AWB_STATE_CONVERGED:
                return AwbState.CONVERGED;
            case CaptureResult.CONTROL_AWB_STATE_LOCKED:
                return AwbState.LOCKED;
            default: // fall out
        }
        Log.e(TAG, "Undefined awb state: " + state);
        return AwbState.UNKNOWN;
    }

    /**
     * Converts the camera2 {@link CaptureResult#FLASH_STATE} to {@link FlashState}.
     *
     * @return the {@link FlashState}.
     */
    @NonNull
    @Override
    public FlashState getFlashState() {
        Integer state = mCaptureResult.get(CaptureResult.FLASH_STATE);
        if (state == null) {
            return FlashState.UNKNOWN;
        }
        switch (state) {
            case CaptureResult.FLASH_STATE_UNAVAILABLE:
            case CaptureResult.FLASH_STATE_CHARGING:
                return FlashState.NONE;
            case CaptureResult.FLASH_STATE_READY:
                return FlashState.READY;
            case CaptureResult.FLASH_STATE_FIRED:
            case CaptureResult.FLASH_STATE_PARTIAL:
                return FlashState.FIRED;
            default: // fall out
        }
        Log.e(TAG, "Undefined flash state: " + state);
        return FlashState.UNKNOWN;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public long getTimestamp() {
        Long timestamp = mCaptureResult.get(CaptureResult.SENSOR_TIMESTAMP);
        if (timestamp == null) {
            return -1L;
        }

        return timestamp;
    }

    @Override
    public Object getTag() {
        return mTag;
    }


    public CaptureResult getCaptureResult() {
        return mCaptureResult;
    }
}

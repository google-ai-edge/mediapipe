/*
 * Copyright 2020 The Android Open Source Project
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

package com.ola.olamera.camerax;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.os.Build;
import android.util.Log;
import android.util.Size;

import androidx.annotation.MainThread;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.annotation.RequiresPermission;
import androidx.camera.core.Camera;
import androidx.camera.core.UseCaseGroup;
import androidx.camera.core.impl.utils.Threads;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.lifecycle.LifecycleOwner;

@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
final class LifecycleCameraController extends CameraController {

    private static final String TAG = "CamLifecycleController";

    @Nullable
    private LifecycleOwner mLifecycleOwner;

    public LifecycleCameraController(@NonNull Context context, Size previewSize) {
        super(context, previewSize);
    }

    public LifecycleCameraController(@NonNull Context context, Size previewSize, boolean isLimitCaptureSize, Size maxCaptureSize, OnCameraProviderListener listener) {
        super(context, previewSize, isLimitCaptureSize, maxCaptureSize, listener);
    }

    @SuppressLint({"MissingPermission", "RestrictedApi"})
    @MainThread
    public void bindToLifecycle(@NonNull LifecycleOwner lifecycleOwner) {
        Threads.checkMainThread();
        mLifecycleOwner = lifecycleOwner;
        startCameraAndTrackStates();
    }

    /**
     * Clears the previously set {@link LifecycleOwner} and stops the camera.
     *
     * @see ProcessCameraProvider#unbindAll
     */
    @SuppressLint("RestrictedApi")
    @MainThread
    public void unbind() {
        Threads.checkMainThread();
        mLifecycleOwner = null;
        mCamera = null;
        if (mCameraProvider != null) {
            mCameraProvider.unbindAll();
        }
    }

    /**
     * Unbind and rebind all use cases to {@link LifecycleOwner}.
     *
     * @return null if failed to start camera.
     */
    @SuppressLint({"UnsafeExperimentalUsageError", "UnsafeOptInUsageError", "RestrictedApi"})
    @RequiresPermission(Manifest.permission.CAMERA)
    @Override
    @Nullable
    Camera startCamera() {
        if (mLifecycleOwner == null) {
            Log.d(TAG, "Lifecycle is not set.");
            return null;
        }
        if (mCameraProvider == null) {
            Log.d(TAG, "CameraProvider is not ready.");
            return null;
        }

        UseCaseGroup useCaseGroup = createUseCaseGroup();
        if (useCaseGroup == null) {
            // Use cases can't be created.
            return null;
        }

        return mCameraProvider.bindToLifecycle(mLifecycleOwner, mCameraSelector, useCaseGroup);
    }

}

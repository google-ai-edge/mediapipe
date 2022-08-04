package com.ola.olamera.camera.camera;
/*
 *
 *  Creation    :  2021/3/29
 *  Author      : jiaming.wjm@
 */

import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.os.Build;
import android.os.Handler;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

import static android.hardware.camera2.CameraAccessException.CAMERA_ERROR;

@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)

public class CameraManagerCompatApi28Impl implements CameraManagerCompatImpl {
    @Override
    public void openCamera(@NonNull CameraManager cameraManager, @NonNull String cameraId, @NonNull Handler handler, @NonNull CameraDevice.StateCallback callback) throws CameraAccessException {
        try {
            // Pass through directly to the executor API that exists on this API level.
            cameraManager.openCamera(cameraId, callback, handler);
        } catch (CameraAccessException e) {
            throw e;
        } catch (IllegalArgumentException | SecurityException e) {
            // Re-throw those RuntimeException will be thrown by CameraManager#openCamera.
            throw e;
        } catch (RuntimeException e) {
            if (isDndFailCase(e)) {
                throwDndException(e);
            }
            throw e;
        }
    }

    @NonNull
    @Override
    public CameraCharacteristics getCameraCharacteristics(@NonNull CameraManager cameraManager, @NonNull String cameraId) throws CameraAccessException {
        try {
            return cameraManager.getCameraCharacteristics(cameraId);
        } catch (RuntimeException e) {
            if (isDndFailCase(e)) {
                // No need to get from cache here because we always get the instance from cache in
                // CameraManagerCompat.  So when DnDFail happens, it happens only when it gets
                // the CameraCharacteristics for the first time.
                throwDndException(e);
            }
        }

        return null;
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    private void throwDndException(@NonNull Throwable cause) throws CameraAccessException {
        throw new CameraAccessException(CAMERA_ERROR, "CAMERA_UNAVAILABLE_DO_NOT_DISTURB", cause);
    }


    /*
     * Check if the exception is due to Do Not Disturb being on, which is only on specific builds
     * of P. See b/149413835 and b/132362603.
     */
    private boolean isDndFailCase(@NonNull Throwable throwable) {
        return Build.VERSION.SDK_INT == 28 && isDndRuntimeException(throwable);
    }


    /*
     * The full stack
     *
     * java.lang.RuntimeException: Camera is being used after Camera.release() was called
     *  at android.hardware.Camera._enableShutterSound(Native Method)
     *  at android.hardware.Camera.updateAppOpsPlayAudio(Camera.java:1770)
     *  at android.hardware.Camera.initAppOps(Camera.java:582)
     *  at android.hardware.Camera.<init>(Camera.java:575)
     *  at android.hardware.Camera.getEmptyParameters(Camera.java:2130)
     *  at android.hardware.camera2.legacy.LegacyMetadataMapper.createCharacteristics
     *  (LegacyMetadataMapper.java:151)
     *  at android.hardware.camera2.CameraManager.getCameraCharacteristics(CameraManager.java:274)
     *
     * This method check the first stack is "_enableShutterSound"
     */
    private static boolean isDndRuntimeException(@NonNull Throwable throwable) {
        if (throwable.getClass().equals(RuntimeException.class)) {
            StackTraceElement[] stackTraceElement;
            if ((stackTraceElement = throwable.getStackTrace()) == null
                    || stackTraceElement.length < 0) {
                return false;
            }
            return "_enableShutterSound".equals(stackTraceElement[0].getMethodName());
        }
        return false;
    }
}

package com.ola.olamera.util;
/*
 *
 *  Creation    :  20-11-19
 *  Author      : jiaming.wjm@
 */

import android.hardware.camera2.CameraDevice;
import android.os.Build;

import androidx.annotation.RequiresApi;


public class CameraLogger {

    public interface ILogger {
        void onError(String tag, String message);

        void onInfo(String tag, String message);

        void onTestLongLog(String tag, String message);

        void uploadError(String tag, String message);
    }

    private static ILogger sLogger;

    public static void setLoggerImp(ILogger logger) {
        sLogger = logger;
    }



    public static void e(String tag, String message, Object... other) {
        try {

            if (sLogger == null) {
                return;
            }

            if (other != null) {
                message = String.format(message, other);
            }

            sLogger.onError(tag, message);
        } catch (Exception e) {
        }
    }


    public static void i(String tag, String message, Object... other) {
        try {

            if (sLogger == null) {
                return;
            }

            if (other != null) {
                message = String.format(message, other);
            }

            sLogger.onInfo(tag, message);

        } catch (Exception e) {
            CameraShould.fail(message, e);
        }
    }

    /**
     * 通用debug开启
     */
    @TestOnly
    public static void testLongLog(String tag, String message, Object... other) {
        try {
            if (sLogger == null) {
                return;
            }
            message = String.format(message, other);

            sLogger.onTestLongLog(tag, message);

        } catch (Exception e) {
            CameraShould.fail(message, e);
        }
    }

    public static void uploadError(String tag, String message) {
        try {
            if (sLogger == null) {
                return;
            }

            sLogger.uploadError(tag, message);

        } catch (Exception e) {
            CameraShould.fail(message, e);
        }
    }


    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    @SuppressWarnings("WeakerAccess") /* synthetic accessor */
    public static String getCameraErrorMessage(int errorCode) {
        switch (errorCode) {
//            case Camera2CameraImpl.ERROR_NONE:
//                return "ERROR_NONE";
            case CameraDevice.StateCallback.ERROR_CAMERA_DEVICE:
                return "ERROR_CAMERA_DEVICE";
            case CameraDevice.StateCallback.ERROR_CAMERA_DISABLED:
                return "ERROR_CAMERA_DISABLED";
            case CameraDevice.StateCallback.ERROR_CAMERA_IN_USE:
                return "ERROR_CAMERA_IN_USE";
            case CameraDevice.StateCallback.ERROR_CAMERA_SERVICE:
                return "ERROR_CAMERA_SERVICE";
            case CameraDevice.StateCallback.ERROR_MAX_CAMERAS_IN_USE:
                return "ERROR_MAX_CAMERAS_IN_USE";
            case ERROR_TRY_REOPEN_ERROR:
                return "ERROR_TRY_REOPEN_ERROR_OVER_MAX_TIMES";
            default: // fall out
        }
        return "UNKNOWN ERROR ( " + errorCode + " ) ";
    }

    public static final int ERROR_TRY_REOPEN_ERROR = -10;


}

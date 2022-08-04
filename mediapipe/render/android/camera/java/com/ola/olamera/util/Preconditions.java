package com.ola.olamera.util;


import androidx.annotation.RestrictTo;

//@RestrictTo(RestrictTo.Scope.LIBRARY_GROUP)
public class Preconditions {


    public static long CAMERA_THREAD_ID = -1;

    public static void setCameraThread(long id) {
        CAMERA_THREAD_ID = id;
    }

    public static void checkState(boolean state) {
        if (!state) {
            CameraShould.fail("");
        }
    }

    public static void checkState(boolean state, String message) {
        if (!state) {
            CameraShould.fail(message);
        }
    }

    public static void onException(Exception e) {
        CameraShould.fail("", e);
    }


    public static void cameraThreadCheck() {
        if (CAMERA_THREAD_ID != -1) {
            if (Thread.currentThread().getId() != CAMERA_THREAD_ID) {
                CameraShould.fail("");
            }
        }
    }
}

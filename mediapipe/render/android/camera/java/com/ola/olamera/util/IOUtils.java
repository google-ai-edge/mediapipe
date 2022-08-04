package com.ola.olamera.util;
/*
 *
 *  Creation    :  2021/2/20
 *  Author      : jiaming.wjm@
 */

import androidx.annotation.RestrictTo;

import java.io.Closeable;

@RestrictTo(RestrictTo.Scope.LIBRARY_GROUP)
public class IOUtils {
    public static void safeClose(Closeable closeable) {
        if (closeable == null) {
            return;
        }
        try {
            closeable.close();
        } catch (Exception ignore) {
            CameraShould.fail();
        }
    }
}

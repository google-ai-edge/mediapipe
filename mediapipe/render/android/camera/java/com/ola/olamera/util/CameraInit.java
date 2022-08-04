package com.ola.olamera.util;
/*
 *
 *  Creation    :  2021/4/15
 *  Author      : jiaming.wjm@
 */

import androidx.annotation.NonNull;

public class CameraInit {


    private static CameraEvnConfig sConfig = defaultConfig();

    private static CameraEvnConfig defaultConfig() {
        return new CameraEvnConfig()
                .setDebuggable(false);
    }

    public static class CameraEvnConfig {
        private boolean mDebuggable = false;
        private boolean mEnableHighMemoryGC = true;

        public CameraEvnConfig setDebuggable(boolean debuggable) {
            mDebuggable = debuggable;
            return CameraEvnConfig.this;
        }

        public boolean isDebuggable() {
            return mDebuggable;
        }

        public boolean enableHighMemoryGC() {
            return mEnableHighMemoryGC;
        }

        public CameraEvnConfig setEnableHighMemoryGC(boolean enableHighMemoryGC) {
            mEnableHighMemoryGC = enableHighMemoryGC;
            return CameraEvnConfig.this;
        }
    }

    public static void init(@NonNull CameraEvnConfig config) {
        sConfig = config;
    }

    public static CameraEvnConfig getConfig() {
        return sConfig;
    }
}

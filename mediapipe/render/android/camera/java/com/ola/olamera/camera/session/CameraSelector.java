package com.ola.olamera.camera.session;
/*
 *
 *  Creation    :  20-11-18
 *  Author      : jiaming.wjm@
 */

import java.util.List;

public class CameraSelector implements ISelector {

    @Override
    public List<String> filter(List<String> cameraIds) {
        return null;
    }

    private CameraLenFacing mFacing;


    public enum CameraLenFacing {
        LEN_FACING_FONT("font"),
        LEN_FACING_BACK("back");

        private final String mName;

        CameraLenFacing(String name) {
            this.mName = name;
        }

        public String getName() {
            return mName;
        }
    }

    public CameraSelector setFacing(CameraLenFacing facing) {
        mFacing = facing;
        return this;
    }

    private boolean mUseWideCamera;

    public CameraSelector useWideCamera(boolean useWideCamera) {
        mUseWideCamera = useWideCamera;
        return this;
    }

    public boolean isUseWideCamera() {
        return mUseWideCamera;
    }

    public CameraLenFacing getFacing() {
        return mFacing;
    }

    public CameraSelector() {
    }


    public static CameraSelector from(CameraLenFacing facing) {
        return new CameraSelector().setFacing(facing);
    }

    public String toLogString() {
        if (mFacing != null) {
            return mFacing.getName();
        }
        return "unknown";
    }


    public static class Build {

    }


}

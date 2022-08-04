package com.ola.olamera.camera.concurrent;

import com.ola.olamera.util.SequentialExecutor;

public class CameraExecutors {

    private static SequentialExecutor sCameraTaskExecutor;

    private static SequentialExecutor sImageAnalyzeSubscriptHandler;

    public synchronized static SequentialExecutor getCameraTaskExecutor() {
        return sCameraTaskExecutor;
    }


    public static synchronized void init() {
        //do nothing
        sCameraTaskExecutor = new SequentialExecutor("camera_executor");
        sCameraTaskExecutor.start();

        sImageAnalyzeSubscriptHandler = new SequentialExecutor("image_analyze");
        sImageAnalyzeSubscriptHandler.start();
    }

    public synchronized static SequentialExecutor getImageAnalyzeSubscriptHandler() {
        return sImageAnalyzeSubscriptHandler;
    }


    public static synchronized void unInit() {
        if (sCameraTaskExecutor != null) {
            sCameraTaskExecutor.stop();
            sCameraTaskExecutor = null;
        }
        if (sImageAnalyzeSubscriptHandler != null) {
            sImageAnalyzeSubscriptHandler.stop();
            sImageAnalyzeSubscriptHandler = null;
        }
    }


}

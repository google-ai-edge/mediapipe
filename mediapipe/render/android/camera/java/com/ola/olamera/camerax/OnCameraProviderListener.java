package com.ola.olamera.camerax;

import androidx.camera.lifecycle.ProcessCameraProvider;

/**
 * 在CameraController中Camera创建完成后的回调接口
 *
 * @author : liujian
 * @date : 2021/7/30
 */
public interface OnCameraProviderListener {

    /**
     * ProcessCameraProvider对象创建完成后的回调方法
     *
     * @param provider ProcessCameraProvider
     */
    void onProcessCameraProvider(ProcessCameraProvider provider);

}

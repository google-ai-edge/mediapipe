package com.ola.olamera.camera.preview;
/*
 *
 *  Creation    :  2021/7/12
 *  Author      : jiaming.wjm@
 */

import android.graphics.RectF;
import android.util.Rational;
import android.util.Size;

import androidx.camera.core.FocusMeteringResult;

import com.google.common.util.concurrent.ListenableFuture;
import com.ola.olamera.render.photo.SnapShotCommand;

public interface IPreviewView {

    void doTakePhotoAnimation();

    int getViewRotation();

    @ViewPort.ScaleType
    int getScaleType();

    @Deprecated
    Rational getAspectRatio();

    int getViewHeight();

    int getViewWidth();

    /**
     * 相机画面显示的区域，这个区域可能不是塞满整一个屏幕的
     * 夸克的布局方式比较复杂，无法仅仅通过 scaleType 知道相机渲染的位置
     */
    RectF getCameraShowRect();

    void updateCameraSurfaceSize(Size size);

    Size getCameraSurfaceSize();

    IPreviewSurfaceProvider getSurfaceProvider();

    void snapshot(final SnapShotCommand snapShotCommand);

    ListenableFuture<FocusMeteringResult> autoFocus(float x, float y, float size, long autoCancelTime /*ms*/);
}

package com.ola.olamera.camerax.controller;

/**
 * 点击对焦的事件回调
 *
 * @author : liujian
 * @date : 2021/8/4
 */
public interface OnGestureDetectorListener {

    void onClickFocused(float x, float y);

    void onPinchToZoom(float zoomRatio, float maxZoomRatio, float minZoomRatio);
}

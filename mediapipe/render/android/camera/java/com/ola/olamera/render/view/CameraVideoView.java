package com.ola.olamera.render.view;


import android.annotation.SuppressLint;
import android.content.Context;
import android.os.Build;
import android.util.AttributeSet;
import android.view.Surface;

import com.google.common.util.concurrent.ListenableFuture;
import com.ola.olamera.camera.preview.IPreviewSurfaceProvider;
import com.ola.olamera.camera.preview.SurfaceTextureWrapper;
import com.ola.olamera.render.DefaultCameraRender;
import com.ola.olamera.util.CameraLogger;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.camera.core.FocusMeteringResult;
import androidx.camera.core.impl.utils.futures.Futures;

/**
 * Camera2预览View
 */
@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
public class CameraVideoView extends BasePreviewView {

    public CameraVideoView(@NonNull Context context, @Nullable AttributeSet attrs) {
        this(context, false);
    }

    public CameraVideoView(Context context, boolean ZOrderOverlay) {
        super(context, ZOrderOverlay);
    }

    @Override
    public IPreviewSurfaceProvider getSurfaceProvider() {
        return mPreviewSurfaceProvider;
    }

    @SuppressLint("RestrictedApi")
    @Override
    public ListenableFuture<FocusMeteringResult> autoFocus(float x, float y, float size, long autoCancelTime) {
        return Futures.immediateFailedFuture(new Throwable("not support"));
    }

    private DefaultCameraRender mDefaultCameraRender;
    private final Object mSurfaceLock = new Object();

    private final IPreviewSurfaceProvider mPreviewSurfaceProvider = new IPreviewSurfaceProvider() {
        @Override
        public Surface provide(@NonNull SurfaceRequest request) {
            synchronized (mSurfaceLock) {
                //释放当前surface texture
                onUseComplete(null);

                CameraLogger.i("CameraLifeManager", "onSurfaceRequested: previewWidth:" + request.width + "  height:" + request.height);

                SurfaceTextureWrapper surfaceTextureWrapper = new SurfaceTextureWrapper(request.width, request.height);
                mDefaultCameraRender = new DefaultCameraRender(surfaceTextureWrapper, request.camera2Camera, request.repeatCaptureCallback);
                getRender().setCameraRender(mDefaultCameraRender);

                return surfaceTextureWrapper.getSurface();
            }
        }

        @Override
        public void onUseComplete(Surface surface) {
            synchronized (mSurfaceLock) {
                if (mDefaultCameraRender != null) {
                    CameraLogger.i("CameraVideoView", "onReleaseUseComplete");
                    mDefaultCameraRender.destroySurface();
                    mDefaultCameraRender = null;
                }
            }
        }
    };
}
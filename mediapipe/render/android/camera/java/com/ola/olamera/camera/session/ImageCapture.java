package com.ola.olamera.camera.session;
/*
 *
 *  Creation    :  2021/4/19
 *  Author      : jiaming.wjm@
 */

import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.RectF;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.media.Image;
import android.os.Build;
import android.util.LayoutDirection;
import android.util.Size;

import com.ola.olamera.camera.camera.Camera2CameraImpl;
import com.ola.olamera.camera.camera.Camera2Info;
import com.ola.olamera.camera.imagereader.DeferrableImageReader;
import com.ola.olamera.camera.preview.IPreviewView;
import com.ola.olamera.camera.preview.ViewPorts;
import com.ola.olamera.camera.session.config.CameraConfigUtils;
import com.ola.olamera.util.CameraLogger;
import com.ola.olamera.util.ImageUtils;
import com.ola.olamera.util.Should;

import java.io.ByteArrayInputStream;
import java.nio.ByteBuffer;
import java.util.LinkedList;
import java.util.Locale;
import java.util.Queue;

import androidx.annotation.IntRange;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.exifinterface.media.ExifInterface;

import static com.ola.olamera.util.ImageUtils.min;
import static com.ola.olamera.util.ImageUtils.sizeToVertexes;

@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
public class ImageCapture {

    private DeferrableImageReader mDeferrableImageReader;

    private Camera2CameraImpl mCamera;

    private IPreviewView mPreviewView;

    public static final class CameraCaptureLogBean {
        public boolean isCameraFocused = false;
        public boolean isCameraZoomed = false;
    }

    /**
     * Callback for when an image capture has completed.
     */
    public abstract static class OnImageCapturedCallback {

        private RectF mCameraShowRectAccordingToPreviewView;
        // 相机是否聚焦过，用于埋点上报
        private CameraCaptureLogBean mCameraLogBean;

        public RectF getCameraShowRectAccordingToPreviewView() {
            return mCameraShowRectAccordingToPreviewView;
        }

        public void setCameraShowRectAccordingToPreviewView(RectF cameraShowRectAccordingToPreviewView) {
            mCameraShowRectAccordingToPreviewView = cameraShowRectAccordingToPreviewView;
        }

        public CameraCaptureLogBean getCameraLogBean() {
            return mCameraLogBean;
        }

        public void setCameraLogBean(CameraCaptureLogBean cameraLogBean) {
            this.mCameraLogBean = cameraLogBean;
        }

        public abstract void onCaptureSuccess(@NonNull byte[] jpeg, Size size, Rect previewCropRect, int rotation);

        public void onError(@NonNull final Exception exception) {
        }


    }

    public void bindPreviewView(IPreviewView previewView) {
        mPreviewView = previewView;
    }

    public void bindCamera(Camera2CameraImpl control) {
        mCamera = control;
    }


    private Size mCaptureSurfaceSize;

    public void updateCaptureSurfaceSize(Size size) {
        mCaptureSurfaceSize = size;
    }


    /**
     * 预览画面相对于拍照内容的区域
     */
    public Rect getViewPointRect() {
        if (mPreviewView == null || mCamera == null || mCaptureSurfaceSize == null) {
            return null;
        }

        Camera2Info camera2Info = mCamera.getCamera2Info();

        return ViewPorts.calculateViewPortRect(
                camera2Info.getSensorRect(),
                camera2Info.isFrontCamera(),
                mPreviewView.getAspectRatio(),
                getRelativeRotation(mPreviewView.getViewRotation()),
                mPreviewView.getScaleType(),
                LayoutDirection.LTR,
                mCaptureSurfaceSize
        );
    }

    public RectF getPreviewShowRectAccordingToView() {
        if (mPreviewView == null || mCamera == null) {
            return null;
        }
        return mPreviewView.getCameraShowRect();
    }

    @IntRange(from = 0, to = 359)
    private int getRelativeRotation(int previewRotation) {
        if (previewRotation != 0) {
            Should.fail("not support now");
        }
        if (mCamera != null) {
            return mCamera.getCamera2Info().getSensorOrientation();
        }
        return 0;
    }


    private static class CaptureTask {
        private final int mRelativeDegrees;
        private final @NonNull
        SingleCaptureConfig mConfig;
        private final Rect mViewPortRect;
        private final @NonNull
        ImageCapture.OnImageCapturedCallback mCallback;

        public CaptureTask(@Nullable Rect viewPortCropRect,
                           int relativeRotation,
                           @NonNull SingleCaptureConfig config,
                           @NonNull OnImageCapturedCallback callback,
                           @Nullable RectF cameraPreviewShowRectAccordingToView) {
            this.mConfig = config;
            this.mViewPortRect = viewPortCropRect;
            this.mRelativeDegrees = relativeRotation;
            this.mCallback = callback;
            mCallback.setCameraShowRectAccordingToPreviewView(cameraPreviewShowRectAccordingToView);
        }

        public @NonNull
        SingleCaptureConfig getConfig() {
            return mConfig;
        }

        public void dispatchImage(Image image) {
            int dispatchRotationDegrees = 0;

            try {
                Size size = new Size(image.getWidth(), image.getHeight());
                Image.Plane[] planes = image.getPlanes();
                ByteBuffer buffer = planes[0].getBuffer();
                byte[] data = new byte[buffer.capacity()];
                buffer.get(data);

                Size dispatchResolution;
                if (ImageUtils.shouldUseExifOrientation(image)) {
                    try {

                        ExifInterface exif;


                        exif = new ExifInterface(new ByteArrayInputStream(data));
                        buffer.rewind();

                        int exifRotation = exif.getAttributeInt(
                                ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED);

                        dispatchRotationDegrees = ImageUtils.getRotation(exifRotation);
                        int jpegWidth = exif.getAttributeInt(ExifInterface.TAG_IMAGE_WIDTH, 0);
                        int jpegHeight = exif.getAttributeInt(ExifInterface.TAG_IMAGE_LENGTH, 0);

                        dispatchResolution = new Size(jpegWidth, jpegHeight);
                    } catch (Exception e) {
                        notifyError(e);
                        return;
                    }
                } else {
                    dispatchResolution = new Size(image.getWidth(), image.getHeight());
                }
                Rect imageCropRect = getDispatchCropRect(mViewPortRect, mRelativeDegrees, dispatchResolution, dispatchRotationDegrees);
                notifySuccess(data, size, imageCropRect, dispatchRotationDegrees);

            } catch (Exception e) {
                notifyError(e);
            } finally {
                image.close();
            }

        }


        public void notifySuccess(@NonNull byte[] jpeg, Size size, Rect previewCropRect, int rotation) {
            mCallback.onCaptureSuccess(jpeg, size, previewCropRect, rotation);
        }

        public void notifyError(Exception throwable) {
            mCallback.onError(throwable);
        }


    }


    private final Queue<CaptureTask> mPendingTasks = new LinkedList<>();
    private CaptureTask mRunningTask;

    private void addTask(@NonNull CaptureTask task) {
        mPendingTasks.add(task);
    }

    private void triggerNext() {
        if (mRunningTask != null) {
            return;
        }
        mRunningTask = mPendingTasks.poll();

        if (mRunningTask == null) {
            return;
        }
        if (mCamera == null) {
            mRunningTask.notifyError(new IllegalStateException("not camera bind"));
            triggerNext();
        }

        mCamera.capture(mRunningTask.getConfig(), new InnerImageCaptureCallback() {
            @Override
            public void onCaptureStart() {
                if (mPreviewView != null) {
                    mPreviewView.doTakePhotoAnimation();
                }
            }

            @Override
            public void onCaptureSuccess(Image image) {
                mRunningTask.dispatchImage(image);
                mRunningTask = null;
                triggerNext();
            }

            @Override
            public void onError(Exception e) {
                mRunningTask.notifyError(e);
                mRunningTask = null;
                triggerNext();
            }
        });
    }


    public void takePicture(@NonNull SingleCaptureConfig singleCaptureConfig,
                            @NonNull ImageCapture.OnImageCapturedCallback capturedCallback) {


        Rect viewPortCropRect = getViewPointRect();
        RectF previewShowRectAccordingToView = getPreviewShowRectAccordingToView();
        CameraLogger.i("ImageCapture", "update capture view port crop rect %s", viewPortCropRect);
        addTask(new CaptureTask(
                viewPortCropRect,
                mPreviewView != null ? getRelativeRotation(mPreviewView.getViewRotation()) : 0,
                singleCaptureConfig,
                capturedCallback,
                previewShowRectAccordingToView));

        triggerNext();
    }


    public void setDeferrableImageReader(DeferrableImageReader deferrableImageReader) {
        mDeferrableImageReader = deferrableImageReader;
    }


    public DeferrableImageReader getDeferrableImageReader() {
        return mDeferrableImageReader;
    }


    void fillConfig(@NonNull CameraCharacteristics cameraCharacteristics, @NonNull CaptureRequest.Builder builder) {
        // Use the same AE and AF modes as the preview.
        //AF:自动对焦
        CameraConfigUtils.checkAndSetConfigIntValue(cameraCharacteristics, builder,
                CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES, CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_PICTURE);

        //AE:自动曝光
        CameraConfigUtils.checkAndSetConfigIntValue(cameraCharacteristics, builder,
                CameraCharacteristics.CONTROL_AE_AVAILABLE_MODES, CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON);

    }

    /**
     * Corrects crop rect based on JPEG exif rotation.
     *
     * <p> The original crop rect is calculated based on camera sensor buffer. On some devices,
     * the buffer is rotated before being passed to users, in which case the crop rect also
     * needs additional transformations.
     *
     * <p> There are two most common scenarios: 1) exif rotation is 0, or 2) exif rotation
     * equals output rotation. 1) means the HAL rotated the buffer based on target
     * rotation. 2) means HAL no-oped on the rotation. Theoretically only 1) needs
     * additional transformations, but this method is also generic enough to handle all possible
     * HAL rotations.
     */
    @NonNull
    static Rect getDispatchCropRect(@NonNull Rect surfaceCropRect, int surfaceToOutputDegrees,
                                    @NonNull Size dispatchResolution, int dispatchToOutputDegrees) {


        CameraLogger.i("ViewPorts", String.format(Locale.CHINA, "getDispatchCropRect surfaceCropRect:%s " +
                        "surfaceToOutputDegrees:%d dispatchResolution:%s dispatchToOutputDegrees:%d", surfaceCropRect,
                surfaceToOutputDegrees, dispatchResolution, dispatchToOutputDegrees));


        // There are 3 coordinate systems: surface, dispatch and output. Surface is where
        // the original crop rect is defined. We need to figure out what HAL
        // has done to the buffer (the surface->dispatch mapping) and apply the same
        // transformation to the crop rect.
        // The surface->dispatch mapping is calculated by inverting a dispatch->surface mapping.

        Matrix matrix = new Matrix();
        // Apply the dispatch->surface rotation.
        matrix.setRotate(dispatchToOutputDegrees - surfaceToOutputDegrees);
        // Apply the dispatch->surface translation. The translation is calculated by
        // compensating for the offset caused by the dispatch->surface rotation.
        float[] vertexes = sizeToVertexes(dispatchResolution);
        matrix.mapPoints(vertexes);
        float left = min(vertexes[0], vertexes[2], vertexes[4], vertexes[6]);
        float top = min(vertexes[1], vertexes[3], vertexes[5], vertexes[7]);
        matrix.postTranslate(-left, -top);
        // Inverting the dispatch->surface mapping to get the surface->dispatch mapping.
        matrix.invert(matrix);

        // Apply the surface->dispatch mapping to surface crop rect.
        RectF dispatchCropRectF = new RectF();
        matrix.mapRect(dispatchCropRectF, new RectF(surfaceCropRect));
        dispatchCropRectF.sort();
        Rect dispatchCropRect = new Rect();
        dispatchCropRectF.round(dispatchCropRect);
        return dispatchCropRect;
    }


}

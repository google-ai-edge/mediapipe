package com.ola.olamera.camera.session;


import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.os.Build;
import android.util.Range;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

import com.ola.olamera.camera.camera.Camera2Info;
import com.ola.olamera.camera.camera.CameraSurfaceHelper;
import com.ola.olamera.camera.imagereader.DeferrableImageReader;
import com.ola.olamera.camera.preview.IPreviewView;
import com.ola.olamera.camera.session.config.CameraConfigUtils;

import java.util.List;
import java.util.concurrent.Executor;

@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
public class PreviewConfig {

    private final int mExceptWidth;
    private final int mExceptHeight;

    private int mActualWidth;
    private int mActualHeight;

    private RepeatCaptureRequestConfig mPreviewCaptureConfig;


    private CameraSurfaceHelper.ISuggestionCalculation mSuggestionCalculation;

    private @NonNull
    IPreviewView mPreviewView;

    private List<DeferrableImageReader> mImageReaders;

    private final CameraCaptureComboCallback mRepeatCaptureCallback = new CameraCaptureComboCallback();


    public CameraCaptureComboCallback getRepeatCaptureCallback() {
        return mRepeatCaptureCallback;
    }

    public PreviewConfig setImageReaders(List<DeferrableImageReader> imageReaders) {
        mImageReaders = imageReaders;
        return this;
    }


    public PreviewConfig setSizeCalculation(CameraSurfaceHelper.ISuggestionCalculation sizeCalculation) {
        mSuggestionCalculation = sizeCalculation;
        return this;
    }

    public CameraSurfaceHelper.ISuggestionCalculation getSuggestionCalculation() {
        return mSuggestionCalculation;
    }

    public List<DeferrableImageReader> getImageReaders() {
        return mImageReaders;
    }

    /**
     * 相机传感器的宽高,默认情况下，相机传感器长边为宽
     * 最终给出的Surface的宽高不一定等于预期的，会从硬件支持中，找到最相近的尺寸
     *
     * @param exceptWidth  　相机传感器预期宽度（长边）
     * @param exceptHeight 　相机传感器预期高度（短边）
     */
    public PreviewConfig(int exceptWidth, int exceptHeight, @NonNull IPreviewView previewView) {
        mExceptWidth = exceptWidth;
        mExceptHeight = exceptHeight;
        mPreviewView = previewView;
    }

    public IPreviewView getPreviewView() {
        return mPreviewView;
    }


    public int getExpectWidth() {
        return mExceptWidth;
    }

    public int getExpectHeight() {
        return mExceptHeight;
    }

    public int getActualWidth() {
        return mActualWidth;
    }

    public void setActualWidth(int actualWidth) {
        mActualWidth = actualWidth;
    }

    public int getActualHeight() {
        return mActualHeight;
    }

    public void setActualHeight(int actualHeight) {
        mActualHeight = actualHeight;
    }


    public @NonNull
    synchronized RepeatCaptureRequestConfig getRepeatCaptureRequestConfig() {
        if (mPreviewCaptureConfig == null) {
            mPreviewCaptureConfig = new RepeatCaptureRequestConfig() {

                @Override
                public void fillConfig(@NonNull Camera2Info cameraInfo, @NonNull CaptureRequest.Builder builder) {
                    CameraCharacteristics cameraCharacteristics = cameraInfo.getCameraCharacteristics();
//                    //预览的默认配置
//
                    Range<Integer> ae_compensation_range =
                            cameraCharacteristics.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE);

                    int expectAECompensation = 0;
                    if (ae_compensation_range.contains(expectAECompensation)) {
                        builder.set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, expectAECompensation);
                        builder.set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, expectAECompensation);
                    }


                    //AF:自动对焦

                    CameraConfigUtils.checkAndSetConfigIntValue(cameraCharacteristics, builder,
                            CameraCharacteristics.CONTROL_AE_AVAILABLE_MODES, CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON);


                    //AF:自动对焦

                    CameraConfigUtils.checkAndSetConfigIntValue(cameraCharacteristics, builder,
                            CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES, CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_PICTURE);


                    //AWB:自动白平衡


                    CameraConfigUtils.checkAndSetConfigIntValue(cameraCharacteristics, builder,
                            CameraCharacteristics.CONTROL_AWB_AVAILABLE_MODES, CaptureRequest.CONTROL_AWB_MODE, CameraMetadata.CONTROL_AWB_MODE_AUTO);


                    if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M) {
                        CameraConfigUtils.checkAndSetConfigIntValue(cameraCharacteristics, builder,
                                CameraCharacteristics.CONTROL_AVAILABLE_MODES, CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
                    }

//                    if (cameraInfo.getCameraLenFacing() == CameraSelector.CameraLenFacing.LEN_FACING_BACK) {
//                        //临时做法，临时在代码设计，后续将前后相机的配置分开实现
//                        //前置相机的AE时间要长一点，不然画面会偏暗
//                        Range<Integer>[] fpsRanges = cameraCharacteristics.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
//
//                        Range<Integer> bestRange = null;
//                        Range<Integer> secondRange = null;
//                        for (Range<Integer> range : fpsRanges) {
//                            if (range.getLower() == 30 && range.getUpper() == 30) {
//                                bestRange = range;
//                                break;
//                            }
//                            if (range.getLower() >= 25 && range.getUpper() <= 35) {
//                                secondRange = range;
//                            }
//                        }
//                        if (bestRange == null) {
//                            bestRange = secondRange;
//                        }
//
//                        if (bestRange != null) {
//                            builder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, bestRange);
//                        }
//                    }

                    builder.setTag("Preview&ImageReader");

                }

                @Override
                public synchronized CameraCaptureCallback getCallback() {
                    return mRepeatCaptureCallback;
                }

                @Override
                public synchronized Executor getCallbackExecutor() {
                    return null;
                }
            }

            ;
        }
        return mPreviewCaptureConfig;
    }
}

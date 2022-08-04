package com.ola.olamera.camera.camera;


import android.graphics.Rect;
import android.hardware.camera2.CameraCharacteristics;
import android.os.Build;
import android.util.Log;
import android.util.Range;
import android.util.Size;
import android.util.SizeF;

import com.ola.olamera.camera.sensor.DisplayOrientationDetector;
import com.ola.olamera.camera.sensor.ImageRotationHelper;
import com.ola.olamera.camera.session.CameraSelector;
import com.ola.olamera.util.CameraInit;
import com.ola.olamera.util.CameraLogger;
import com.ola.olamera.util.CameraShould;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;


@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
public class Camera2Info {
    // 最大的水平广角 超过认为是超广角 可能会导致变形
    private static final int MAX_ANGLE = 90;

    private CameraCharacteristics mCameraCharacteristics;

    private final String mCameraId;

    private boolean mIsFrontCamera;

    private ImageRotationHelper mImageRotationHelper;

    private Rect mSensorRect;

    public Camera2Info(@NonNull String cameraId) {
        mCameraId = cameraId;
    }

    public void setCameraCharacteristics(@NonNull CameraCharacteristics cameraCharacteristics) {
        mCameraCharacteristics = cameraCharacteristics;
        initCommonInfo();
        calculateFOV();
        printCameraInfo();
    }

    private void initCommonInfo() {
        try {
            mIsFrontCamera =
                    mCameraCharacteristics.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_FRONT;
            mSensorRect = mCameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);
        } catch (Exception e) {
            mSensorRect = null;
        }
    }

    public boolean isFrontCamera() {
        return mIsFrontCamera;
    }

    public Rect getSensorRect() {
        return mSensorRect;
    }

    public void setDisplayOrientationDetector(DisplayOrientationDetector detector) {
        mImageRotationHelper = new ImageRotationHelper(this, detector);
    }

    public ImageRotationHelper getImageRotationHelper() {
        return mImageRotationHelper;
    }

    private void printCameraInfo() {
        if (!CameraInit.getConfig().isDebuggable()) {
            return;
        }

        CameraLogger.i("CameraLifeManager", getCameraInfoMessage());
    }

    /**
     * active_array_aspect = ACTIVE_ARRAY.w / ACTIVE_ARRAY.h
     * output_a_aspect = output_a.w / output_a.h
     * output_b_aspect = output_b.w / output_b.h
     * output_a_physical_height = SENSOR_INFO_PHYSICAL_SIZE.y * ACTIVE_ARRAY.h / PIXEL_ARRAY.h * output_a_aspect / active_array_aspect
     * output_b_physical_height = SENSOR_INFO_PHYSICAL_SIZE.y * ACTIVE_ARRAY.h / PIXEL_ARRAY.h * output_b_aspect / active_array_aspect
     * FOV_a_y = 2 * atan(output_a_physical_height / (2 * LENS_FOCAL_LENGTH))
     * FOV_b_y = 2 * atan(output_b_physical_height / (2 * LENS_FOCAL_LENGTH))
     *
     * @return
     */


    public CameraCharacteristics getCameraCharacteristics() {
        return mCameraCharacteristics;
    }

    public int getSensorOrientation() {
        Integer sensorOrientation =
                mCameraCharacteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
        return sensorOrientation != null ? sensorOrientation : 0;
    }

    public CameraSelector.CameraLenFacing getCameraLenFacing() {
        Integer lens_facing = mCameraCharacteristics.get(CameraCharacteristics.LENS_FACING);
        if (lens_facing == null) {
            return null;
        }
        if (lens_facing == CameraCharacteristics.LENS_FACING_FRONT) {
            return CameraSelector.CameraLenFacing.LEN_FACING_FONT;
        } else if (lens_facing == CameraCharacteristics.LENS_FACING_BACK) {
            return CameraSelector.CameraLenFacing.LEN_FACING_BACK;
        }
        return null;
    }

    public static final float INVALID_FOCAL_LENGTH = Float.MAX_VALUE;

    public static class FocalLengthInfo implements Comparable<FocalLengthInfo> {
        public float focalLength = INVALID_FOCAL_LENGTH;
        public double horizontalAngle = 0; //0 ~ pi
        public double verticalAngle = 0;  //0 ~ pi
        public boolean isDefaultFocal = false;

        @Override
        public int compareTo(FocalLengthInfo o) {
            if (o == this) {
                return 0;
            }
            //never happen
            if (focalLength == o.focalLength) {
                return 0;
            }

            return focalLength > o.focalLength ? 1 : -1;
        }
    }

    public String getCameraId() {
        return mCameraId;
    }

    private final List<FocalLengthInfo> mFocalLengthInfos = new ArrayList<>();


    /**
     * https://stackoverflow.com/questions/39965408/what-is-the-android-camera2-api-equivalent-of-camera-parameters-gethorizontalvie
     * <p>
     * FOV.x = 2 * atan(SENSOR_INFO_PHYSICAL_SIZE.x / (2 * LENS_FOCAL_LENGTH))
     * FOV.y = 2 * atan(SENSOR_INFO_PHYSICAL_SIZE.y / (2 * LENS_FOCAL_LENGTH))
     * <p>
     * <p>
     * ignore sensor_pixel info and capture crop region first  @(- 3 -)
     */
    private void calculateFOV() {

        mFocalLengthInfos.clear();

        try {

            SizeF size = mCameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE);

            float[] focalLens = mCameraCharacteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS);

            double width = size.getWidth();

            double height = size.getHeight();


            boolean firstOne = true;
            for (float focalLength : focalLens) {
                FocalLengthInfo lengthInfo = new FocalLengthInfo();
                lengthInfo.horizontalAngle = (2 * Math.atan(width / (focalLength * 2)));
                lengthInfo.verticalAngle = (2 * Math.atan(height / (focalLength * 2)));
                lengthInfo.focalLength = focalLength;
                //一般情况下，第一个焦距信息为默认焦距
                lengthInfo.isDefaultFocal = firstOne;
                firstOne = false;
                mFocalLengthInfos.add(lengthInfo);
            }

            Collections.sort(mFocalLengthInfos);


        } catch (Exception e) {
            CameraShould.fail("", e);
        }

    }


    public List<FocalLengthInfo> getFocalLengthInfos() {
        return mFocalLengthInfos;
    }

    /**
     * 返回最佳的焦距信息
     */
    public FocalLengthInfo getBestFocalInfo() {
        for (FocalLengthInfo info : mFocalLengthInfos) {
            double angle = info.horizontalAngle * 180 / Math.PI;
            if (angle <= MAX_ANGLE) {
                return info;
            }
        }
        // 如果所有焦距对应角度都大于90 选择默认焦距
        return getDefaultFocalInfo();
    }

    /**
     * 一般而言，就是默认的焦距
     */
    public FocalLengthInfo getDefaultFocalInfo() {
        for (FocalLengthInfo info : mFocalLengthInfos) {
            if (info.isDefaultFocal) {
                return info;
            }
        }
        return null;
    }

    /**
     * 输出当前物理镜头焦距 广角等信息
     */
    public String getCameraInfoMessage() {
        Integer support_level = mCameraCharacteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);

        StringBuilder cameraInfo = new StringBuilder(" \n{\n");
        cameraInfo.append("cameraId:").append(mCameraId).append(",\nsupport_level:").append(support_level);
        cameraInfo.append("\n");
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            Range<Float> zoom_range = mCameraCharacteristics.get(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE);
            cameraInfo.append("zoom_range:[").append(zoom_range.getLower()).append(":").append(zoom_range.getUpper()).append("]").append(",\n");
        }

        float max_digital_zoom = mCameraCharacteristics.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM);
        cameraInfo.append("max_digital_zoom:").append(max_digital_zoom).append(",");
        cameraInfo.append("\n");

        SizeF sensorPhysicalSize = mCameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE);

        cameraInfo.append(String.format(Locale.CHINA, "physical: %.2f * %.2f ", sensorPhysicalSize.getWidth(), sensorPhysicalSize.getHeight())).append(",\n");

        Size sensorPixelSize = mCameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE);

        cameraInfo.append(String.format(Locale.CHINA, "pixel: %d * %d ", sensorPixelSize.getWidth(), sensorPixelSize.getHeight())).append(",\n");

        //active pixel info log
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
            int[] distortion_modes =
                    mCameraCharacteristics.get(CameraCharacteristics.DISTORTION_CORRECTION_AVAILABLE_MODES);

            cameraInfo.append("distortion_correction_modes : [");
            if (distortion_modes != null) {
                for (int value : distortion_modes) {
                    cameraInfo.append(value).append(",");
                }
            } else {
                cameraInfo.append("null");
            }
            cameraInfo.append("]");
            cameraInfo.append("\n");
        }

        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M) {
            Rect preCorrectionActiveArraySize =
                    mCameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE);
            cameraInfo.append(String.format(Locale.CHINA, "pre_correction_active_array_size: %s ", preCorrectionActiveArraySize)).append(",\n");
        }

        Rect sensorActiveArrayRect = mCameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);

        cameraInfo.append(String.format(Locale.CHINA, "active_rect: %s ", sensorActiveArrayRect)).append(",\n");

        cameraInfo.append("focal_info {");
        for (FocalLengthInfo focalLengthInfo : mFocalLengthInfos) {
            float active_array_aspect = ((float) sensorActiveArrayRect.width()) / sensorActiveArrayRect.height();
            float output_a_aspect = 3648f / 2736f;
            float output_a_physical_height = sensorPhysicalSize.getHeight() * sensorActiveArrayRect.height() / sensorPixelSize.getHeight() * output_a_aspect / active_array_aspect;
            float fov_a_y = (float) (2 * Math.atan(output_a_physical_height / (2 * focalLengthInfo.focalLength)));

            Log.e("CameraLife", "active_array_aspect: " + active_array_aspect);
            cameraInfo.append("\n   ").append(String.format(Locale.CHINA, "foca:  l_length:%.2f  h_angle:%.2f    " +
                            "v_angle:%.2f   fov_a_y:%.2f ",
                    focalLengthInfo.focalLength, focalLengthInfo.horizontalAngle / Math.PI * 180,
                    focalLengthInfo.verticalAngle / Math.PI * 180, fov_a_y / Math.PI * 180)).append(",");
        }
        cameraInfo.append("\n}\n");
        cameraInfo.append("}");
        return cameraInfo.toString();
    }

    public String getSelectCameraInfoMsg() {
        StringBuilder cameraInfo = new StringBuilder();
        // 当前选择的镜头焦距信息
        FocalLengthInfo bestFocalInfo = getBestFocalInfo();
        if (bestFocalInfo != null) {
            cameraInfo.append("{")
                    .append("\n ").append(String.format(Locale.CHINA, "l_length:%.2f, \nh_angle:%.2f,\n" +
                            "v_angle:%.2f",
                    bestFocalInfo.focalLength, bestFocalInfo.horizontalAngle / Math.PI * 180,
                    bestFocalInfo.verticalAngle / Math.PI * 180));
            cameraInfo.append("}");
        }
        return cameraInfo.toString();
    }
}

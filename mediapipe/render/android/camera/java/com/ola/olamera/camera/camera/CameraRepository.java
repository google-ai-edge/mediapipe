package com.ola.olamera.camera.camera;
/*
 *
 *  Creation    :  20-11-23
 *  Author      : jiaming.wjm@
 */

import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.os.Build;
import android.os.Handler;
import android.text.TextUtils;
import android.util.Log;
import android.util.Size;

import com.ola.olamera.camera.anotaion.ExecutedBy;
import com.ola.olamera.camera.session.CameraSelector;
import com.ola.olamera.util.CameraLogger;
import com.ola.olamera.util.CameraShould;
import com.ola.olamera.util.Preconditions;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Executor;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

/**
 * 管理所有CameraID对应的Camera实例的仓库
 */
@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
public class CameraRepository {

    HashMap<String, Camera2CameraImpl> mCameras = new HashMap<>();


    public CameraRepository() {
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    @ExecutedBy("CameraExecutor")
    public synchronized void init(@NonNull CameraManager cameraManager, Executor executor, Handler handler) {
        Preconditions.cameraThreadCheck();
        try {
            mCameras.clear();
            String[] cameraList = cameraManager.getCameraIdList();
            for (String cameraId : cameraList) {
                CameraCharacteristics characteristics =
                        CameraManagerCompatImpl.from().getCameraCharacteristics(cameraManager, cameraId);
                Camera2CameraImpl camera2Camera = new Camera2CameraImpl(cameraManager, cameraId, executor, handler);
                camera2Camera.setCameraCharacteristics(characteristics);
                mCameras.put(cameraId, camera2Camera);
            }
        } catch (CameraAccessException e) {
            CameraShould.fail("", e);
            CameraLogger.e("CameraRepository", "init camera error \n %s", Log.getStackTraceString(e));
        }
    }

    public synchronized List<Camera2CameraImpl> filterCamera(@NonNull CameraSelector selector) {

        boolean isFrontCamera = selector.getFacing() == CameraSelector.CameraLenFacing.LEN_FACING_FONT;
        String defaultCameraId = null;
        List<Camera2CameraImpl> matchCameraList = new ArrayList<>();
        for (Map.Entry<String, Camera2CameraImpl> entry : mCameras.entrySet()) {
            try {
                Camera2CameraImpl camera2Camera = entry.getValue();
                CameraCharacteristics characteristics = camera2Camera.getCameraCharacteristics();

                Integer lens_facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (lens_facing == null) {
                    continue;
                }

                StreamConfigurationMap streamConfigurationMap =
                        characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

                Size[] sizes = streamConfigurationMap.getOutputSizes(SurfaceTexture.class);

                //必须要支持预览，像TOF摄像头, 仅仅支持输出DEPTH_16的数据
                if (sizes == null || sizes.length == 0) {
                    continue;
                }

                if (isFrontCamera) {
                    if (lens_facing == CameraCharacteristics.LENS_FACING_FRONT) {
                        matchCameraList.add(entry.getValue());
                    }
                } else {
                    if (lens_facing == CameraCharacteristics.LENS_FACING_BACK) {
                        matchCameraList.add(entry.getValue());
                    }
                }
                //相机ID最小，或者是第一个相机为默认镜头，前置一般为0，后置一般为1
                if (defaultCameraId == null) {
                    defaultCameraId = entry.getValue().getCameraId();
                }
            } catch (Exception e) {
                CameraShould.fail("", e);
            }
        }

        //使用广角优先摄像头
        if (selector.isUseWideCamera()) {
            //Collections.sort(matchCameraList, new DefaultFocalHorizontalFOVComparator(defaultCameraId));
        }

        //if (CameraInit.getConfig().isDebuggable()) {
        StringBuilder sb = new StringBuilder("\n");
        for (Camera2CameraImpl camera2Camera : matchCameraList) {
            Camera2Info.FocalLengthInfo minFL = camera2Camera.getCamera2Info().getBestFocalInfo();
            sb.append(String.format(Locale.CHINA, "camera (%s) min_fl: %.2f ; max_h_angle : %.2f ",
                    camera2Camera.getCameraId(),
                    (minFL != null ? minFL.focalLength : -1),
                    (minFL != null ? minFL.horizontalAngle * 180 / Math.PI : -1)))
                    .append("\n");
        }
        CameraLogger.i("CameraLifeManager", "print match camera list\n %s", sb);

        //}


        return matchCameraList;
    }


    private static class HorizontalFOVComparator implements Comparator<Camera2CameraImpl> {

        @Override
        public int compare(Camera2CameraImpl o1, Camera2CameraImpl o2) {
            if (o1 == o2) {
                return 0;
            }

            //将空值放后面去
            if (o1.getCamera2Info().getBestFocalInfo() == null) {
                return 1;
            }
            if (o2.getCamera2Info().getBestFocalInfo() == null) {
                return -1;
            }

            if (o1.getCamera2Info().getBestFocalInfo().horizontalAngle
                    == o2.getCamera2Info().getBestFocalInfo().horizontalAngle) {

                //相同水平FOV的情况下，比较camera_id，经验而言,camera_id比较小是主镜头
                return o1.getCameraId().compareTo(o2.getCameraId()) > 0 ? 1 : -1;
            }

            //水平FOV越大，约靠前
            return o1.getCamera2Info().getBestFocalInfo().horizontalAngle >
                    o2.getCamera2Info().getBestFocalInfo().horizontalAngle ? -1 : 1;
        }

    }

    private static class DefaultFocalHorizontalFOVComparator implements Comparator<Camera2CameraImpl> {

        private final String mDefaultCameraId;

        public DefaultFocalHorizontalFOVComparator(String defaultCameraId) {
            mDefaultCameraId = defaultCameraId;
        }

        @Override
        public int compare(Camera2CameraImpl o1, Camera2CameraImpl o2) {
            if (o1 == o2) {
                return 0;
            }

            //将空值放后面去
            if (o1.getCamera2Info().getDefaultFocalInfo() == null) {
                return 1;
            }
            if (o2.getCamera2Info().getDefaultFocalInfo() == null) {
                return -1;
            }

            if (o1.getCamera2Info().getDefaultFocalInfo().horizontalAngle
                    == o2.getCamera2Info().getDefaultFocalInfo().horizontalAngle) {

                //相同FOV的情况下，将默认镜头往后排，因为我们优先希望使用广角摄像头，使用其光圈等参数设置，而不单单是FOV
                if (TextUtils.equals(o1.getCameraId(), mDefaultCameraId)) {
                    return 1;
                }

                if (TextUtils.equals(o2.getCameraId(), mDefaultCameraId)) {
                    return -1;
                }

                //相同水平FOV的情况下，比较camera_id，经验而言,camera_id比较小是主镜头
                return o1.getCameraId().compareTo(o2.getCameraId()) > 0 ? 1 : -1;
            }

            //水平FOV越大，约靠前
            return o1.getCamera2Info().getDefaultFocalInfo().horizontalAngle >
                    o2.getCamera2Info().getDefaultFocalInfo().horizontalAngle ? -1 : 1;
        }

    }

    public static void cameraListAutoTest() {

    }


}

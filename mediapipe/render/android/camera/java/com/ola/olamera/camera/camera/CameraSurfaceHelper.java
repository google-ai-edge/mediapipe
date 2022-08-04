package com.ola.olamera.camera.camera;
/*
 *
 *  Creation    :  20-11-23
 *  Author      : jiaming.wjm@
 */

import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.os.Build;
import android.util.Log;
import android.util.Rational;
import android.util.Size;
import android.view.SurfaceHolder;

import com.ola.olamera.camera.imagereader.DeferrableImageReader;
import com.ola.olamera.camera.sensor.ImageRotationHelper;
import com.ola.olamera.camera.session.ImageCapture;
import com.ola.olamera.camera.session.PreviewConfig;
import com.ola.olamera.util.CameraInit;
import com.ola.olamera.util.CameraLogger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

@RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
public class CameraSurfaceHelper {


    //相机传感器默认给的图像是横的,所以最大值也是横着来设置
    public static final Size MAX_SIZE = new Size(1920, 1080);


    public static void configPreviewSize(PreviewConfig config, Camera2CameraImpl camera2Camera) throws Exception {

        CameraCharacteristics characteristics = camera2Camera.getCameraCharacteristics();
        StreamConfigurationMap streamConfigurationMap = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

        int[] supportFormats = streamConfigurationMap.getOutputFormats();

        for (int value : supportFormats) {
            Log.e("wujm-camera", "support out format : " + Integer.toHexString(value));
        }


        Size actualPreviewSize = CameraSurfaceHelper.getSurfaceTextureSuggestionSize(
                new Size(config.getExpectWidth(), config.getExpectHeight()),
                camera2Camera,
                config.getSuggestionCalculation() != null ?
                        config.getSuggestionCalculation() : LimitedClosestSizeCalculation.getInstance());


        if (actualPreviewSize == null) {
            throw new IllegalArgumentException(String.format("not match preview size for (%d,%d)", config.getExpectWidth(), config.getExpectHeight()));
        }

        config.setActualHeight(actualPreviewSize.getHeight());
        config.setActualWidth(actualPreviewSize.getWidth());


        CameraLogger.i("CameraLifeManager",
                "find preview suggestion size (%d,%d) from expect (%d,%d) ",
                config.getActualWidth(), config.getActualHeight(), config.getExpectWidth(), config.getExpectHeight());
    }

    public static void configImageCaptureSize(ImageCapture config, Camera2CameraImpl camera2Camera, ImageRotationHelper rotationHelper) {
        DeferrableImageReader imageReader = config.getDeferrableImageReader();
        configImageReader(Collections.singletonList(imageReader), camera2Camera, rotationHelper, config.getDeferrableImageReader().getSizeCalculation());
        config.updateCaptureSurfaceSize(new Size(imageReader.getActualWidth(), imageReader.getActualHeight()));
    }

    public static void configImageReader(List<DeferrableImageReader> imageReaders,
                                         Camera2CameraImpl camera2Camera,
                                         ImageRotationHelper rotationHelper) {
        configImageReader(imageReaders, camera2Camera, rotationHelper, LimitedClosestSizeCalculation.getInstance());
    }

    public static void configImageReader(List<DeferrableImageReader> imageReaders,
                                         Camera2CameraImpl camera2Camera,
                                         ImageRotationHelper rotationHelper,
                                         ISuggestionCalculation calculation) {
        if (imageReaders == null) {
            return;
        }

        CameraCharacteristics characteristics = camera2Camera.getCameraCharacteristics();
        StreamConfigurationMap streamConfigurationMap = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        if (streamConfigurationMap == null) {
            return;
        }

        int[] supportFormats = streamConfigurationMap.getOutputFormats();

        Arrays.sort(supportFormats);

        StringBuilder sb = new StringBuilder("Camera SupportFormat ( ");
        for (int format : supportFormats) {
            sb.append("0x").append(Integer.toHexString(format)).append(" ");
        }
        sb.append(")");
        CameraLogger.i("CameraLifeManager", sb.toString());


        for (DeferrableImageReader imageReader : imageReaders) {

            int index = Arrays.binarySearch(supportFormats, imageReader.getFormat());

            if (index == -1) {
                CameraLogger.e("CameraLifeManager", "not support image reader format 0x%s", Integer.toHexString(imageReader.getFormat()));
                continue;
            }

            Size bestSize = getSurfaceSuggestionSize(imageReader, camera2Camera, calculation);
            if (bestSize == null) {
                continue;
            }

            imageReader.createAndroidImageReader(
                    bestSize.getWidth(), bestSize.getHeight(),
                    rotationHelper);


            CameraLogger.i("CameraLifeManager",
                    "find image reader suggestion size (%d,%d) from expect (%d,%d) in format ",
                    bestSize.getWidth(), bestSize.getHeight(), imageReader.getExpectWidth(), imageReader.getExpectHeight(), imageReader.getFormat());
        }
    }


    public static Size getSurfaceTextureSuggestionSize(Size previewSize, Camera2CameraImpl camera2Camera, @NonNull ISuggestionCalculation calculation) {

        CameraCharacteristics characteristics = camera2Camera.getCameraCharacteristics();
        StreamConfigurationMap streamConfigurationMap = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        if (streamConfigurationMap == null) {
            return null;
        }
        Size[] sizes = streamConfigurationMap.getOutputSizes(SurfaceTexture.class);

        if (CameraInit.getConfig().isDebuggable()) {
            StringBuilder sb = new StringBuilder("[");
            for (Size size : sizes) {
                sb.append(" (").append(size.getWidth()).append("*").append(size.getHeight()).append(") ,");
            }
            sb.append("]");
            CameraLogger.i("CameraLifeManager", "surface size Support : %s", sb);
        }
        Rect senorRect = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);
        Size senorSize = characteristics.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE);
        return calculation.getSuggestionSize(senorSize, senorRect, sizes, previewSize.getWidth(),
                previewSize.getHeight());
    }

    public static Size getSurfaceViewSuggestionSize(Size previewSize, Camera2CameraImpl camera2Camera, @NonNull ISuggestionCalculation calculation) {

        CameraCharacteristics characteristics = camera2Camera.getCameraCharacteristics();
        StreamConfigurationMap streamConfigurationMap = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        if (streamConfigurationMap == null) {
            return null;
        }
        Rect senorRect = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);
        Size senorSize = characteristics.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE);
        Size[] sizes = streamConfigurationMap.getOutputSizes(SurfaceHolder.class);

        return calculation.getSuggestionSize(senorSize, senorRect, sizes, previewSize.getWidth(),
                previewSize.getHeight());
    }

    public static Size getSurfaceSuggestionSize(@NonNull DeferrableImageReader imageReader, Camera2CameraImpl camera2Camera, @NonNull ISuggestionCalculation calculation) {

        CameraCharacteristics characteristics = camera2Camera.getCameraCharacteristics();
        StreamConfigurationMap streamConfigurationMap = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        if (streamConfigurationMap == null) {
            return null;
        }
        Rect senorRect = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);
        Size senorSize = characteristics.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE);

        Size[] sizes = streamConfigurationMap.getOutputSizes(imageReader.getFormat());
        if (CameraInit.getConfig().isDebuggable()) {
            StringBuilder sb = new StringBuilder("[");
            for (Size size : sizes) {
                sb.append(" (").append(size.getWidth()).append("*").append(size.getHeight()).append(") ,");
            }
            sb.append("]");
            CameraLogger.i("CameraLifeManager", "image reader size Support : %s", sb);
        }

        return calculation.getSuggestionSize(senorSize, senorRect, sizes, imageReader.getExpectWidth(),
                imageReader.getExpectHeight());
    }

    public interface ISuggestionCalculation {
        public Size getSuggestionSize(Size sensorSize, Rect sensorRect, Size[] sizes, int expectWidth,
                                      int expectHeight);
    }

    /**
     * 限制最大 {@link #MAX_SIZE} 的预览大小，寻找最接近
     */
    public static class LimitedClosestSizeCalculation implements ISuggestionCalculation {

        @Override
        public Size getSuggestionSize(Size sensorSize, Rect sensorRect, Size[] sizes, int expectWidth, int expectHeight) {
            if (sizes == null) {
                return null;
            }

            int minDiff = Integer.MAX_VALUE;
            Size bestSize = null;
            for (Size size : sizes) {
                //最大清晰度不超过1080
                if (getArea(size) > 1.1 * getArea(MAX_SIZE)) {
                    continue;
                }
                int curDiff = Math.abs(size.getWidth() - expectWidth) + Math.abs(size.getHeight() - expectHeight);
                if (curDiff < minDiff) {
                    minDiff = curDiff;
                    bestSize = size;
                }
            }

            return bestSize;
        }

        private static LimitedClosestSizeCalculation sInstance;

        public synchronized static LimitedClosestSizeCalculation getInstance() {
            if (sInstance == null) {
                sInstance = new LimitedClosestSizeCalculation();
            }
            return sInstance;
        }
    }

    public static class AreaClosestSizeCalculation implements ISuggestionCalculation {

        @Override
        public Size getSuggestionSize(Size sensorSize, Rect sensorRect, Size[] sizes, int expectWidth, int expectHeight) {
            if (sizes == null) {
                return null;
            }

            int minDiff = Integer.MAX_VALUE;
            Size bestSize = null;
            for (Size size : sizes) {
                int curDiff = Math.abs(size.getWidth() - expectWidth) + Math.abs(size.getHeight() - expectHeight);
                if (curDiff < minDiff) {
                    minDiff = curDiff;
                    bestSize = size;
                }
            }
            return bestSize;
        }

        private static AreaClosestSizeCalculation sInstance;

        public synchronized static AreaClosestSizeCalculation getInstance() {
            if (sInstance == null) {
                sInstance = new AreaClosestSizeCalculation();
            }
            return sInstance;
        }
    }

    public static class MaxAreaCalculation implements ISuggestionCalculation {

        @Override
        public Size getSuggestionSize(Size sensorSize, Rect sensorRect, Size[] sizes, int expectWidth,
                                      int expectHeight) {
            long cur_size_area = 0;
            Size bestSize = null;
            for (Size size : sizes) {
                long size_area = size.getHeight() * size.getHeight();
                if (size_area > cur_size_area) {
                    cur_size_area = size_area;
                    bestSize = size;
                }
            }
            return bestSize;
        }

        private static MaxAreaCalculation sInstance;

        public synchronized static MaxAreaCalculation getInstance() {
            if (sInstance == null) {
                sInstance = new MaxAreaCalculation();
            }
            return sInstance;
        }
    }

    public static class MaxSensorAspectCalculation implements ISuggestionCalculation {

        @Override
        public Size getSuggestionSize(Size sensorSize, Rect sensorRect, Size[] sizes, int expectWidth, int expectHeight) {


            long cur_size_area = 0;
            Size maxSize = null;
            for (Size size : sizes) {
                long size_area = ((long) size.getWidth()) * size.getHeight();
                if (size_area > cur_size_area) {
                    cur_size_area = size_area;
                    maxSize = size;
                }
            }

            float max_with_height_aspect = maxSize != null ? ((float) maxSize.getWidth() / maxSize.getHeight()) : -1;

            /**
             * 将预期的宽高转换成传感器aspect的宽高
             */
            int aspect_height = expectHeight;
            int aspect_width = (int) (max_with_height_aspect * aspect_height);
            return CameraSurfaceHelper.AreaClosestSizeCalculation.getInstance().getSuggestionSize(sensorSize, sensorRect, sizes, aspect_width, aspect_height);
        }

        private static MaxSensorAspectCalculation sInstance;

        public synchronized static MaxSensorAspectCalculation getInstance() {
            if (sInstance == null) {
                sInstance = new MaxSensorAspectCalculation();
            }
            return sInstance;
        }
    }

    public static int getArea(Size size) {
        return size.getWidth() * size.getHeight();
    }


    public static class AreaAspectSizeCalculation implements ISuggestionCalculation {
        private static final Rational ASPECT_RATIO_4_3 = new Rational(4, 3);
        private static final Rational ASPECT_RATIO_16_9 = new Rational(16, 9);

        public Size getMaxSize() {
            return null;
        }

        public Size getMinSize() {
            return new Size(800, 600);
        }

        @Override
        public Size getSuggestionSize(Size sensorSize, Rect sensorRect, Size[] sizes, int expectWidth, int expectHeight) {
            if (sizes == null) {
                return null;
            }

            // 根据面积排序
            Arrays.sort(sizes, new CompareSizesByArea(true));

            Rational targetRatio = new Rational(expectWidth, expectHeight);
            Size targetSize = new Size(expectWidth, expectHeight);

            Size maxSize = getMaxSize();
            Size minSize = getMinSize();
            if (getArea(targetSize) < getArea(minSize)) {
                minSize = targetSize;
            }
            // 限制最大的Size
            List<Size> outputSizeCandidates = new ArrayList<>();
            for (Size outputSize : sizes) {
                if ((maxSize == null || getArea(outputSize) <= getArea(maxSize)) && getArea(outputSize) >= getArea(minSize)
                        && !outputSizeCandidates.contains(outputSize)) {
                    outputSizeCandidates.add(outputSize);
                }
            }

            // 根据宽高比分组分辨率
            Map<Rational, List<Size>> aspectRatioSizeListMap = groupSizesByAspectRatio(outputSizeCandidates);
            // 每一组删除不必要的较大和较小的Size
            for (Rational key : aspectRatioSizeListMap.keySet()) {
                removeSupportedSizesByTargetSize(aspectRatioSizeListMap.get(key), targetSize);
            }

            // 目标宽高比差值绝对值 小->大 排序
            List<Rational> aspectRatios = new ArrayList<>(aspectRatioSizeListMap.keySet());
            Collections.sort(aspectRatios,
                    new CompareAspectRatiosByDistanceToTargetRatio(targetRatio));

            List<Size> supportedResolutions = new ArrayList<>();
            for (Rational rational : aspectRatios) {
                for (Size size : aspectRatioSizeListMap.get(rational)) {
                    if (!supportedResolutions.contains(size)) {
                        supportedResolutions.add(size);
                    }
                }
            }

            if (supportedResolutions.isEmpty()) {
                return sizes[0];
            }
            CameraLogger.i("CameraLifeManager", "className:" + getClass().getName() + " SuggestionSize: " + supportedResolutions);
            return supportedResolutions.get(0);
        }

        private void removeSupportedSizesByTargetSize(List<Size> supportedSizesList,
                                                      Size targetSize) {
            if (supportedSizesList == null || supportedSizesList.isEmpty()) {
                return;
            }

            int indexBigEnough = -1;
            int indexSmallEnough = -1;

            List<Size> removeSizes = new ArrayList<>();

            for (int i = 0; i < supportedSizesList.size(); i++) {
                Size outputSize = supportedSizesList.get(i);
                if (outputSize.getWidth() >= targetSize.getWidth()
                        && outputSize.getHeight() >= targetSize.getHeight()) {

                    if (indexBigEnough >= 0) {
                        removeSizes.add(supportedSizesList.get(indexBigEnough));
                    }

                    indexBigEnough = i;
                } else {
                    if (indexSmallEnough >= 0) {
                        removeSizes.add(supportedSizesList.get(i));
                    }

                    indexSmallEnough = i;
                }
            }

            supportedSizesList.removeAll(removeSizes);
        }

        private Map<Rational, List<Size>> groupSizesByAspectRatio(List<Size> sizes) {
            Map<Rational, List<Size>> aspectRatioSizeListMap = new HashMap<>();

            aspectRatioSizeListMap.put(ASPECT_RATIO_4_3, new ArrayList<>());
            aspectRatioSizeListMap.put(ASPECT_RATIO_16_9, new ArrayList<>());

            for (Size outputSize : sizes) {
                Rational matchedKey = null;

                for (Rational key : aspectRatioSizeListMap.keySet()) {
                    if (hasMatchingAspectRatio(outputSize, key)) {
                        matchedKey = key;

                        List<Size> sizeList = aspectRatioSizeListMap.get(matchedKey);
                        if (!sizeList.contains(outputSize)) {
                            sizeList.add(outputSize);
                        }
                    }
                }

                if (matchedKey == null) {
                    aspectRatioSizeListMap.put(
                            new Rational(outputSize.getWidth(), outputSize.getHeight()),
                            new ArrayList<>(Collections.singleton(outputSize)));
                }
            }

            return aspectRatioSizeListMap;
        }

        private static final Size DEFAULT_SIZE = new Size(640, 480);
        private static final int ALIGN16 = 16;

        static boolean hasMatchingAspectRatio(Size resolution, Rational aspectRatio) {
            boolean isMatch = false;
            if (aspectRatio == null) {
                isMatch = false;
            } else if (aspectRatio.equals(
                    new Rational(resolution.getWidth(), resolution.getHeight()))) {
                isMatch = true;
            } else if (getArea(resolution) >= getArea(DEFAULT_SIZE)) {
                isMatch = isPossibleMod16FromAspectRatio(resolution, aspectRatio);
            }
            return isMatch;
        }

        private static boolean isPossibleMod16FromAspectRatio(Size resolution, Rational aspectRatio) {
            int width = resolution.getWidth();
            int height = resolution.getHeight();

            Rational invAspectRatio = new Rational(aspectRatio.getDenominator(),
                    aspectRatio.getNumerator());
            if (width % 16 == 0 && height % 16 == 0) {
                return ratioIntersectsMod16Segment(Math.max(0, height - ALIGN16), width, aspectRatio)
                        || ratioIntersectsMod16Segment(Math.max(0, width - ALIGN16), height,
                        invAspectRatio);
            } else if (width % 16 == 0) {
                return ratioIntersectsMod16Segment(height, width, aspectRatio);
            } else if (height % 16 == 0) {
                return ratioIntersectsMod16Segment(width, height, invAspectRatio);
            }
            return false;
        }

        private static boolean ratioIntersectsMod16Segment(int height, int mod16Width,
                                                           Rational aspectRatio) {
            double aspectRatioWidth =
                    height * aspectRatio.getNumerator() / (double) aspectRatio.getDenominator();
            return aspectRatioWidth > Math.max(0, mod16Width - ALIGN16) && aspectRatioWidth < (
                    mod16Width + ALIGN16);
        }
    }

    public static class PreviewAreaAspectCalculation extends AreaAspectSizeCalculation {
        private static final Size PREVIEW_MAX_SIZE = new Size(2000, 1500);
        private static final Size PREVIEW_MIN_SIZE = new Size(800, 600);

        @Override
        public Size getMaxSize() {
            return PREVIEW_MAX_SIZE;
        }
        @Override
        public Size getMinSize() {
            return PREVIEW_MIN_SIZE;
        }

        private static PreviewAreaAspectCalculation sInstance;

        public synchronized static PreviewAreaAspectCalculation getInstance() {
            if (sInstance == null) {
                sInstance = new PreviewAreaAspectCalculation();
            }
            return sInstance;
        }
    }

    public static class CaptureAreaAspectCalculation extends AreaAspectSizeCalculation {
        private static final Size CAPTURE_MAX_SIZE = new Size(4400, 3300);
        private static final Size CAPTURE_MIN_SIZE = new Size(1200, 900);

        @Override
        public Size getMaxSize() {
            return CAPTURE_MAX_SIZE;
        }

        @Override
        public Size getMinSize() {
            return CAPTURE_MIN_SIZE;
        }

        private static CaptureAreaAspectCalculation sInstance;

        public synchronized static CaptureAreaAspectCalculation getInstance() {
            if (sInstance == null) {
                sInstance = new CaptureAreaAspectCalculation();
            }
            return sInstance;
        }
    }

    static final class CompareAspectRatiosByDistanceToTargetRatio implements Comparator<Rational> {
        private Rational mTargetRatio;

        CompareAspectRatiosByDistanceToTargetRatio(Rational targetRatio) {
            mTargetRatio = targetRatio;
        }

        @Override
        public int compare(Rational lhs, Rational rhs) {
            if (lhs.equals(rhs)) {
                return 0;
            }

            final Float lhsRatioDelta = Math.abs(lhs.floatValue() - mTargetRatio.floatValue());
            final Float rhsRatioDelta = Math.abs(rhs.floatValue() - mTargetRatio.floatValue());

            int result = (int) Math.signum(lhsRatioDelta - rhsRatioDelta);
            return result;
        }
    }

    static final class CompareSizesByArea implements Comparator<Size> {
        private boolean mReverse = false;

        CompareSizesByArea() {
        }

        CompareSizesByArea(boolean reverse) {
            mReverse = reverse;
        }

        @Override
        public int compare(Size lhs, Size rhs) {
            int result =
                    Long.signum(
                            (long) lhs.getWidth() * lhs.getHeight()
                                    - (long) rhs.getWidth() * rhs.getHeight());

            if (mReverse) {
                result *= -1;
            }

            return result;
        }
    }

}

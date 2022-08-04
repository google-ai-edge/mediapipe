package com.ola.olamera.util;
/*
 *
 *  Creation    :  2021/7/13
 *  Author      : jiaming.wjm@
 */

import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.RectF;
import android.media.Image;
import android.opengl.Matrix;
import android.os.Build;
import android.util.Log;
import android.util.Rational;
import android.util.Size;

import com.ola.olamera.camera.preview.ViewPort;

import androidx.annotation.IntRange;
import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.exifinterface.media.ExifInterface;

public class ImageUtils {

    /**
     * Converts a {@link Size} to an float array of vertexes.
     */
    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    @NonNull
    public static float[] sizeToVertexes(@NonNull Size size) {
        return new float[]{0, 0, size.getWidth(), 0, size.getWidth(), size.getHeight(), 0,
                size.getHeight()};
    }

    /**
     * Returns the min value.
     */
    public static float min(float value1, float value2, float value3, float value4) {
        return Math.min(Math.min(value1, value2), Math.min(value3, value4));
    }


    /**
     * Checks whether the exif orientation value should be used for the final output image.
     *
     * <p>On some devices, the orientation value in the embedded exif of the captured images may
     * be 0 but the image buffer data actually is not rotated to upright orientation by HAL. For
     * these devices, the exif orientation value should not be used for the final output image.
     *
     * @param image The captured image object.
     */
    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    public static boolean shouldUseExifOrientation(@NonNull Image image) {
        if (image.getFormat() != ImageFormat.JPEG) {
            return false;
        }

        if (isHuaweiMate20Lite() || isHonor9X()) {
            return false;
        }

        return true;
    }

    private static boolean isHuaweiMate20Lite() {
        return "HUAWEI".equalsIgnoreCase(Build.BRAND) && "SNE-LX1".equalsIgnoreCase(Build.MODEL);
    }

    private static boolean isHonor9X() {
        return "HONOR".equalsIgnoreCase(Build.BRAND) && "STK-LX1".equalsIgnoreCase(
                Build.MODEL);
    }


    /**
     * @return The degree of rotation (eg. 0, 90, 180, 270).
     */
    public static int getRotation(int exifRotation) {
        switch (exifRotation) {
            case ExifInterface.ORIENTATION_FLIP_HORIZONTAL:
                return 0;
            case ExifInterface.ORIENTATION_ROTATE_180:
                return 180;
            case ExifInterface.ORIENTATION_FLIP_VERTICAL:
                return 180;
            case ExifInterface.ORIENTATION_TRANSPOSE:
                return 270;
            case ExifInterface.ORIENTATION_ROTATE_90:
                return 90;
            case ExifInterface.ORIENTATION_TRANSVERSE:
                return 90;
            case ExifInterface.ORIENTATION_ROTATE_270:
                return 270;
            case ExifInterface.ORIENTATION_NORMAL:
                // Fall-through
            case ExifInterface.ORIENTATION_UNDEFINED:
                // Fall-through
            default:
                return 0;
        }
    }

    /**
     * Rotates aspect ratio based on rotation degrees.
     */
    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    @NonNull
    public static Rational getRotatedAspectRatio(
            @IntRange(from = 0, to = 359) int rotationDegrees,
            @NonNull Rational aspectRatio) {
        if (rotationDegrees == 90 || rotationDegrees == 270) {
            return inverseRational(aspectRatio);
        }

        return new Rational(aspectRatio.getNumerator(), aspectRatio.getDenominator());
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    private static Rational inverseRational(Rational rational) {
        if (rational == null) {
            return rational;
        }
        return new Rational(
                /*numerator=*/ rational.getDenominator(),
                /*denominator=*/ rational.getNumerator());
    }

    /**
     * 根据scale类型，对顶点坐标矩阵进行转换
     *
     * @param scaleType 目前仅仅支持 {@link ViewPort#FILL_CENTER}
     */
    public static boolean getScaleVertexMatrix(@ViewPort.ScaleType int scaleType, float[] outMatrix,
                                               int in_width, int in_height, int out_width, int out_height) {
        if (in_width <= 0
                || in_height <= 0
                || out_width <= 0
                || out_height <= 0) {
            return false;
        }

        switch (scaleType) {
            case ViewPort.FILL_CENTER: {
                Matrix.setIdentityM(outMatrix, 0);
                float input_radio = (float) in_height / (float) in_width;
                float out_radio = (float) out_height / (float) out_width;

                //注意顶点坐标空间为 -1,1,所以只要直接缩放就行
                if (input_radio > out_radio) {
                    Matrix.scaleM(outMatrix, 0, 1f, (float) input_radio / (float) out_radio, 1.0F);
                } else {
                    Matrix.scaleM(outMatrix, 0, (float) out_radio / (float) input_radio, 1f, 1.0F);
                }
                return true;
            }
            case ViewPort.CENTER_INSIDE: {
                Matrix.setIdentityM(outMatrix, 0);
                float input_radio = (float) in_height / (float) in_width;
                float out_radio = (float) out_height / (float) out_width;

                Log.i("A_TAG", "in_height: " + in_height + " in_width: " + in_width + "  " +
                        "   input_radio: " + input_radio + "  out_height:" + out_height + "  out_width：" + out_width + " out_radio： " + out_radio);
                if (input_radio > out_radio) {
                    Matrix.scaleM(outMatrix, 0, out_radio / input_radio, 1f, 1.0F);
                } else {
                    Matrix.scaleM(outMatrix, 0, 1f, input_radio / out_radio, 1.0F);
                }
                break;
            }
            default: {
                CameraShould.fail("not support now");
            }
        }
        return false;
    }

    /**
     * 注意顶点坐标空间为 -1,1
     *
     * @param heightPercentage 需要裁剪成的Y的高度，0.7表示裁剪掉底部0.3区域
     */
    public static boolean getClipVertexMatrix(float[] outMatrix, float[] marginPercentage) {
        Matrix.setIdentityM(outMatrix, 0);
        Matrix.translateM(outMatrix, 0, 0, marginPercentage[3] - marginPercentage[1], 0.0F);
        Matrix.scaleM(outMatrix, 0, 1f, 1 - marginPercentage[1] - marginPercentage[3], 1.0F);
        return true;
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    public static Rect convert2JpegRect(Rect jpegRect, int jpegRotation, Rect desRect, Size dstSize) {
        RectF jpegShowClip = new RectF(desRect);
        android.graphics.Matrix matrix = new android.graphics.Matrix();
        matrix.postScale(1f / dstSize.getWidth(), 1f / dstSize.getHeight());
        if (jpegRotation != 0) {
            matrix.postRotate(-jpegRotation, 0.5f, 0.5f);
        }

        matrix.postScale(jpegRect.width(), jpegRect.height());
        matrix.postTranslate(jpegRect.left, jpegRect.top);

        matrix.mapRect(jpegShowClip);

        Rect resultJpegShowClip = new Rect();
        jpegShowClip.round(resultJpegShowClip);

        return resultJpegShowClip;
    }


    private final static RectF sPreviewScaleRect = new RectF();
    private final static RectF sCameraSurfaceRect = new RectF();
    private final static RectF sCameraShowRect = new RectF();
    private final static android.graphics.Matrix sSurface2PreviewScaleRectTransform = new android.graphics.Matrix();

    public synchronized static void calculateCameraShowRect(RectF result,
                                                            @ViewPort.ScaleType int scaleType,
                                                            float[] marginPercentage,
                                                            int[] cameraSurfaceSize, int[] windowSize) {


        //        sPreviewScaleRect.set(0, 0,
        //                (1 - marginPercentage[0] - marginPercentage[2]) * windowSize[0],
        //                (1 - marginPercentage[1] - marginPercentage[3]) * windowSize[1]);
        //
        //        sCameraSurfaceRect.set(0, 0, cameraSurfaceSize[0], cameraSurfaceSize[1]);
        //
        //        sSurface2PreviewScaleRectTransform.reset();

        switch (scaleType) {
            case ViewPort.CENTER_INSIDE:
                CameraShould.fail("not support now");
                break;
            case ViewPort.FILL_CENTER:
                result.set(marginPercentage[0] * windowSize[0],
                        marginPercentage[1] * windowSize[1],
                        (1 - marginPercentage[0] - marginPercentage[2]) * windowSize[0],
                        (1 - marginPercentage[1] - marginPercentage[3]) * windowSize[1]
                );
                return;
            default:
                CameraShould.fail("not support now");
        }

        //        sSurface2PreviewScaleRectTransform.mapRect(sCameraShowRect, sCameraSurfaceRect);


    }


}

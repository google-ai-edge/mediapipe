package com.ola.olamera.camerax;

import android.annotation.SuppressLint;
import android.media.ExifInterface;
import android.os.Build;

import java.io.File;
import java.io.IOException;

import androidx.camera.core.impl.utils.Exif;
import androidx.camera.core.internal.compat.workaround.ExifRotationAvailability;

/**
 * 华为一些特殊机型无法正确处理获取旋转信息，exif的旋转方向值错误。对于这些设备，我们应该根据最终输出图像的目标旋转设置计算旋转值
 *
 * @author : yushan.lj@
 * @date : 2022/2/25
 */
public class ExifRotationHelper {


    @SuppressLint("RestrictedApi")
    public static void updateExitRotate(File file, int rotate) {
        if (isNeedUpdateExitInfo()) {
            try {
                Exif exif = Exif.createFromFile(file);

                int orientation = 0;
                switch (rotate) {
                    case 90:
                        orientation = ExifInterface.ORIENTATION_ROTATE_90;
                        break;
                    case 180:
                        orientation = ExifInterface.ORIENTATION_ROTATE_180;
                        break;
                    case 270:
                        orientation = ExifInterface.ORIENTATION_ROTATE_270;
                        break;
                }

                exif.setOrientation(orientation);
                exif.save();
            } catch (IOException e) {
                e.printStackTrace();
            }

        }
    }

    @SuppressLint("RestrictedApi")
    public static boolean isNeedUpdateExitInfo() {
        return !new ExifRotationAvailability().isRotationOptionSupported() || isHuaweiNova3i();
    }

    public static boolean isHuaweiNova3i() {
        return "HUAWEI".equalsIgnoreCase(Build.BRAND) && "INE-TL00".equalsIgnoreCase(Build.MODEL);
    }

}

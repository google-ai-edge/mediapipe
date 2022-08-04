//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by Fernflower decompiler)
//

package com.ola.olamera.util;

import android.opengl.Matrix;
import android.os.Build;
import android.util.Size;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

public class MatrixUtils {
    public static final int TYPE_FITXY = 0;
    public static final int TYPE_CENTERCROP = 1;
    public static final int TYPE_CENTERINSIDE = 2;
    public static final int TYPE_FITSTART = 3;
    public static final int TYPE_FITEND = 4;

    private MatrixUtils() {
    }

    public static void getMatrix(float[] matrix, int type, int imgWidth, int imgHeight, int viewWidth, int viewHeight) {
        if (imgHeight > 0 && imgWidth > 0 && viewWidth > 0 && viewHeight > 0) {
            float[] projection = new float[16];
            float[] camera = new float[16];
            if (type == 0) {
                Matrix.orthoM(projection, 0, -1.0F, 1.0F, -1.0F, 1.0F, 1.0F, 3.0F);
                Matrix.setLookAtM(camera, 0, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F);
                Matrix.multiplyMM(matrix, 0, projection, 0, camera, 0);
                return;
            }

            float sWhView = (float) viewWidth / (float) viewHeight;
            float sWhImg = (float) imgWidth / (float) imgHeight;
            if (sWhImg > sWhView) {
                switch (type) {
                    case 1:
                        Matrix.orthoM(projection, 0, -sWhView / sWhImg, sWhView / sWhImg, -1.0F, 1.0F, 1.0F, 3.0F);
                        break;
                    case 2:
                        Matrix.orthoM(projection, 0, -1.0F, 1.0F, -sWhImg / sWhView, sWhImg / sWhView, 1.0F, 3.0F);
                        break;
                    case 3:
                        Matrix.orthoM(projection, 0, -1.0F, 1.0F, 1.0F - 2.0F * sWhImg / sWhView, 1.0F, 1.0F, 3.0F);
                        break;
                    case 4:
                        Matrix.orthoM(projection, 0, -1.0F, 1.0F, -1.0F, 2.0F * sWhImg / sWhView - 1.0F, 1.0F, 3.0F);
                }
            } else {
                switch (type) {
                    case 1:
                        Matrix.orthoM(projection, 0, -1.0F, 1.0F, -sWhImg / sWhView, sWhImg / sWhView, 1.0F, 3.0F);
                        break;
                    case 2:
                        Matrix.orthoM(projection, 0, -sWhView / sWhImg, sWhView / sWhImg, -1.0F, 1.0F, 1.0F, 3.0F);
                        break;
                    case 3:
                        Matrix.orthoM(projection, 0, -1.0F, 2.0F * sWhView / sWhImg - 1.0F, -1.0F, 1.0F, 1.0F, 3.0F);
                        break;
                    case 4:
                        Matrix.orthoM(projection, 0, 1.0F - 2.0F * sWhView / sWhImg, 1.0F, -1.0F, 1.0F, 1.0F, 3.0F);
                }
            }

            Matrix.setLookAtM(camera, 0, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F);
            Matrix.multiplyMM(matrix, 0, projection, 0, camera, 0);
        }

    }

    public static float[] flip(float[] m, boolean x, boolean y) {
        if (x || y) {
            Matrix.scaleM(m, 0, x ? -1.0F : 1.0F, y ? -1.0F : 1.0F, 1.0F);
        }

        return m;
    }

    public static float[] flipF(float[] matrix, boolean x, boolean y) {
        if (x || y) {
            Matrix.scaleM(matrix, 0, x ? -1.0F : 1.0F, y ? -1.0F : 1.0F, 1.0F);
        }

        return matrix;
    }



}

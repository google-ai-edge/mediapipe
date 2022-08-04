package com.ola.olamera.util;

import android.content.Context;
import android.opengl.Matrix;

import androidx.annotation.RestrictTo;


import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

@RestrictTo(RestrictTo.Scope.LIBRARY_GROUP)
public class OpenGlUtils {


    public static String readRawShaderFile(Context context, String fileName) {

        BufferedReader br = null;
        String line;
        StringBuffer sb = new StringBuffer();
        try {
            // InputStream is = context.getResources().openRawResource(shareId);
            InputStream is = context.getResources().getAssets().open(fileName);

            br = new BufferedReader(new InputStreamReader(is));
            while ((line = br.readLine()) != null) {
                sb.append(line);
                sb.append("\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            IOUtils.safeClose(br);
        }
        return sb.toString();
    }

    public static float[] createIdentityMtx() {
        float[] m = new float[16];
        Matrix.setIdentityM(m, 0);
        return m;
    }

    public static FloatBuffer createSquareVtx() {
        float[] vtx = new float[]{
                -1.0F, 1.0F, 0.0F, 0.0F, 1.0F, -1.0F, -1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 1.0F, 1.0F, 1.0F, -1.0F, 0.0F, 1.0F, 0.0F};
        ByteBuffer bb = ByteBuffer.allocateDirect(4 * vtx.length);
        bb.order(ByteOrder.nativeOrder());
        FloatBuffer fb = bb.asFloatBuffer();
        fb.put(vtx);
        fb.position(0);
        return fb;
    }



}

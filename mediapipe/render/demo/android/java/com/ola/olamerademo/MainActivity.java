package com.ola.olamerademo;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;

import com.ola.olamera.render.view.CameraVideoView;

public class MainActivity extends AppCompatActivity {

    private CameraVideoView mCameraVideoView;
    private ActivityCameraSession mActivityCameraSession;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        mCameraVideoView = new CameraVideoView(this, null);
        setContentView(mCameraVideoView);
        mActivityCameraSession = new ActivityCameraSession(this);
        mActivityCameraSession.setCameraPreview(mCameraVideoView);
        requestPermission(() -> mActivityCameraSession.onWindowCreate());
    }


    private Runnable mPermissionCacheRunnable;


    public static final int REQUEST_CODE = 1234;

    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                if (mPermissionCacheRunnable != null) {
                    mPermissionCacheRunnable.run();
                    mPermissionCacheRunnable = null;
                }
            } else {
                finish();
            }
        }
    }


    @Override
    protected void onResume() {
        super.onResume();

        mActivityCameraSession.onWindowActive();
    }

    @Override
    protected void onPause() {
        super.onPause();
        mActivityCameraSession.onWindowInactive();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mActivityCameraSession.onWindowDestroy();
    }

    /**
     * 请求授权
     */
    private void requestPermission(Runnable runnable) {

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) { //表示未授权时
            //进行授权
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CODE);
            mPermissionCacheRunnable = runnable;
        } else {
            runnable.run();
        }
    }


}
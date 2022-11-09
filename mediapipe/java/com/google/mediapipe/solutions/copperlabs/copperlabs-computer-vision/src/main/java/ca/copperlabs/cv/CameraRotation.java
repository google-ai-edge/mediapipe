package ca.copperlabs.cv;



import android.view.Surface;

public enum CameraRotation {
    FIXED_0_DEG(Surface.ROTATION_0),FIXED_90_DEG(Surface.ROTATION_90),FIXED_180_DEG(Surface.ROTATION_180),FIXED_270_DEG(Surface.ROTATION_270),AUTOMATIC(-1);
    private final  int value;
    private CameraRotation(int rotation) {
        value = rotation;
    }
    public int getValue() {
        return value;
    }
}

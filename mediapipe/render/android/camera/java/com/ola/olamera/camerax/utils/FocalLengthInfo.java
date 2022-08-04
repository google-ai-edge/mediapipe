package com.ola.olamera.camerax.utils;

public class FocalLengthInfo implements Comparable<FocalLengthInfo> {
    public float focalLength = Float.MAX_VALUE;
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

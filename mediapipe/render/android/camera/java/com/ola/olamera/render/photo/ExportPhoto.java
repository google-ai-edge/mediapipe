//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by Fernflower decompiler)
//

package com.ola.olamera.render.photo;

import android.graphics.Bitmap;

import com.ola.olamera.render.entry.FrameDetectData;

import java.util.ArrayList;
import java.util.List;

public class ExportPhoto {
    public int width;
    public int height;
    public byte[] data;
    public ExportPhoto.ImageType dataType;
    public Bitmap bitmap;
    public int rotation;
    private boolean hasClip = false;

    private SnapShotCommand mSnapShotCommand;
    public FrameDetectData mFrameDetectData;

    private final List<ExportPhoto> mSubPhotos = new ArrayList<>();

    public ExportPhoto(int w, int h, byte[] dataIn) {
        this.width = w;
        this.height = h;
        this.data = dataIn;
        this.dataType = ExportPhoto.ImageType.RGBA_8888;
    }

    public void setRotation(int rotation) {
        this.rotation = rotation;
    }

    public ExportPhoto setHasClip(boolean hasClip) {
        this.hasClip = hasClip;
        return this;
    }

    public boolean isHasClip() {
        return this.hasClip;
    }

    public ExportPhoto(int width, int height, byte[] data, ExportPhoto.ImageType imageType) {
        this.width = width;
        this.height = height;
        this.data = data;
        this.dataType = imageType;
    }

    public ExportPhoto(int width, int height, Bitmap bitmap) {
        this.width = width;
        this.height = height;
        this.bitmap = bitmap;
        this.dataType = ExportPhoto.ImageType.BITMAP;
    }

    public static ExportPhoto ofJPEG(byte[] data) {
        if (data == null) {
            return null;
        } else {
            ExportPhoto result = new ExportPhoto(0, 0, data);
            result.dataType = ExportPhoto.ImageType.JPEG_DATA;
            return result;
        }
    }

    public static ExportPhoto ofBitmap(Bitmap bitmap) {
        return bitmap == null ? null : new ExportPhoto(bitmap.getWidth(), bitmap.getHeight(), bitmap);
    }

    public boolean isValid() {
        if (this.dataType == ExportPhoto.ImageType.RGBA_8888) {
            return this.data != null && this.width > 0 && this.height > 0;
        } else if (this.dataType == ExportPhoto.ImageType.JPEG_DATA) {
            return this.data != null;
        } else if (this.dataType == ExportPhoto.ImageType.BITMAP) {
            return this.bitmap != null;
        } else {
            return false;
        }
    }


    public ExportPhoto setFrameDetectData(FrameDetectData frameDetectData) {
        mFrameDetectData = frameDetectData;
        return this;
    }

    public ExportPhoto setSnapShotCommand(SnapShotCommand snapShotCommand) {
        mSnapShotCommand = snapShotCommand;
        return this;
    }

    public SnapShotCommand getSnapShotCommand() {
        return mSnapShotCommand;
    }

    public ExportPhoto addSubPhoto(ExportPhoto photo) {
        mSubPhotos.add(photo);
        return this;
    }

    public List<ExportPhoto> getSubPhotos() {
        return mSubPhotos;
    }


    public static enum ImageType {
        RGBA_8888,
        YUV_NV21,
        JPEG_DATA,
        BITMAP;

        private ImageType() {
        }
    }
}

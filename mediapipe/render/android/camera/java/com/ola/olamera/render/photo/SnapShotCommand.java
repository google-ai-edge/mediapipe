//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by Fernflower decompiler)
//

package com.ola.olamera.render.photo;

import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.RectF;
import android.webkit.ValueCallback;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;

public class SnapShotCommand {
    private Object mTag;
    private float mLeft = 0.0F;
    private float mTop = 0.0F;
    private float mWith = 1.0F;
    private float mHeight = 1.0F;
    private float mScale = 1f;
    private float mRotation = 0;
    private final ValueCallback<ExportPhoto> mCallback;
    private static final Matrix mGLCoordinateMatrix = new Matrix();
    private static final RectF mTempRect = new RectF();

    private final List<SnapShotCommand> mSubCommand = new ArrayList<>();

    public SnapShotCommand(float left, float top, float with, float height, ValueCallback<ExportPhoto> callback) {
        this.mLeft = left;
        this.mTop = top;
        this.mWith = with;
        this.mHeight = height;
        this.mCallback = callback;
    }

    public synchronized SnapShotCommand addSubCommand(SnapShotCommand command) {
        mSubCommand.add(command);
        return this;
    }

    public synchronized List<SnapShotCommand> getSubCommand() {
        return mSubCommand;
    }

    public void set(float left, float top, float with, float height) {
        this.mLeft = left;
        this.mTop = top;
        this.mWith = with;
        this.mHeight = height;
    }

    public SnapShotCommand(ValueCallback<ExportPhoto> callback) {
        this.mCallback = callback;
    }

    public ValueCallback<ExportPhoto> getCallback() {
        return this.mCallback;
    }

    public boolean needClip() {
        if (this.mLeft >= 0.0F && this.mTop >= 0.0F && this.mWith >= 0.0F && this.mWith + this.mLeft <= 1.0F && this.mHeight >= 0.0F && this.mHeight + this.mTop <= 1.0F) {
            return this.mLeft > 0.0F || this.mTop > 0.0F || this.mWith < 1.0F || this.mHeight < 1.0F;
        } else {
            return false;
        }
    }

    public float getScale() {
        return mScale;
    }

    public SnapShotCommand setScale(float scale) {
        mScale = scale;
        return this;
    }

    public void setRotation(float rotation) {
        mRotation = rotation;
    }

    public SnapShotCommand setTag(Object tag) {
        mTag = tag;
        return this;
    }

    public Object getTag() {
        return mTag;
    }

    public Rect getGLSnapshotRect(int pixelWidth, int pixelHeight, boolean needFlipY) {
        if (!this.needClip()) {
            return new Rect(0, 0, pixelWidth, pixelHeight);
        } else if (needFlipY) {
            mGLCoordinateMatrix.reset();
            mGLCoordinateMatrix.postTranslate(-0.5F, -0.5F);
            mGLCoordinateMatrix.postScale(1.0F, -1.0F);
            mGLCoordinateMatrix.postTranslate(0.5F, 0.5F);
            mGLCoordinateMatrix.postScale((float) pixelWidth, (float) pixelHeight);
            mTempRect.set(this.mLeft, this.mTop, this.mWith + this.mLeft, this.mHeight + this.mTop);
            mGLCoordinateMatrix.mapRect(mTempRect);
            return new Rect((int) mTempRect.left, (int) mTempRect.top, (int) mTempRect.right, (int) mTempRect.bottom);
        } else {
            mGLCoordinateMatrix.reset();
            mGLCoordinateMatrix.postScale((float) pixelWidth, (float) pixelHeight);
            mTempRect.set(this.mLeft, this.mTop, this.mWith + this.mLeft, this.mHeight + this.mTop);
            mGLCoordinateMatrix.mapRect(mTempRect);
            return new Rect((int) mTempRect.left, (int) mTempRect.top, (int) mTempRect.right, (int) mTempRect.bottom);
        }
    }
}

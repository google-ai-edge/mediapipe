package com.ola.olamera.camerax.controller;

import androidx.annotation.NonNull;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.MediatorLiveData;

final public class ForwardingLiveData<T> extends MediatorLiveData<T> {

    private LiveData<T> mLiveDataSource;

    public void setSource(@NonNull LiveData<T> liveDataSource) {
        if (mLiveDataSource != null) {
            super.removeSource(mLiveDataSource);
        }
        mLiveDataSource = liveDataSource;
        super.addSource(liveDataSource, this::setValue);
    }

    @Override
    public T getValue() {
        // If MediatorLiveData has no active observers, it will not receive updates
        // when the source is updated, in which case the value of this class and its source
        // will be out-of-sync.
        // We need to retrieve the source value for the caller.
        return mLiveDataSource == null ? null : mLiveDataSource.getValue();
    }
}
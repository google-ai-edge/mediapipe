package com.ola.olamera.render.detector;
/*
 *
 *  Creation    :  2021/1/29
 *  Author      : jiaming.wjm@
 */

public interface IAlgDetector<T> {

    enum State {
        UNINITIALIZED,
        INITIALIZED,
        RUNNING,
    }

    enum InputDataType {
        NV21,
        TEXTURE
    }

    void init();

    void release();

    void start();

    void stop();

    default T detect(IAlgTextureConsumer.NV21Buffer buffer, long timestamp) {
        return null;
    }

    InputDataType getInputDataType();


    State getState();


}

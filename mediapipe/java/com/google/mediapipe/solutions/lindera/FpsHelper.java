package com.google.mediapipe.solutions.lindera;

import java.util.function.Consumer;

public class FpsHelper {
    // 0 no smoothing, 1 is constant output
    double smoothingFactor = 0.9;
    double _fps = -1;
    long startTime = -1;
    public FpsHelper(double smoothingFactor){
        this.smoothingFactor = smoothingFactor;

    }
    public FpsHelper(){}
    public Consumer<Double> onFpsUpdate = null;
    public void logNewPoint(){
        if (startTime==-1){
            startTime = System.nanoTime();

        }
        else{
            long currTime = System.nanoTime();
            long dur = currTime - startTime;
            startTime = currTime;
            double fps = 1e9/dur;
            if (_fps==-1){
                _fps = fps;
            }else {
                _fps = (1 - smoothingFactor) * fps + _fps * smoothingFactor;
            }
            if (onFpsUpdate!=null) {
                onFpsUpdate.accept(_fps);
            }
        }
    }
    public double getFPS(){
        return _fps;
    }



}

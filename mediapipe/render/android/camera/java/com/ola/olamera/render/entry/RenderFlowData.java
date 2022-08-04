//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by Fernflower decompiler)
//

package com.ola.olamera.render.entry;

import java.util.HashMap;
import java.util.LinkedList;

public class RenderFlowData {

    public int texture = -1;
    public int textureWidth;
    public int textureHeight;
    public int windowHeight;
    public int windowWidth;

    public final FrameDetectData detectData = new FrameDetectData();

    public final HashMap<String, Object> extraData = new HashMap<>();


    private RenderFlowData(int windowWidth, int windowHeight) {
        this.windowHeight = windowHeight;
        this.windowWidth = windowWidth;
    }

    public void fillExtraData(RenderFlowData data) {
        data.extraData.putAll(extraData);
    }

    private final static LinkedList<RenderFlowData> sCacheQueue = new LinkedList<>();

    public void recycle() {
        texture = -1;
        textureWidth = 0;
        textureHeight = 0;
        windowHeight = 0;
        windowWidth = 0;
        extraData.clear();
        detectData.points = null;

        synchronized (sCacheQueue) {
            if (sCacheQueue.size() < 100) {
                sCacheQueue.add(this);
            }
        }
    }

    public static RenderFlowData obtain(int windowWidth, int windowHeight, HashMap<String, Object> extra) {
        RenderFlowData data;
        synchronized (sCacheQueue) {
            if (sCacheQueue.size() > 0) {
                data = sCacheQueue.remove();
                data.windowHeight = windowHeight;
                data.windowWidth = windowWidth;
            } else {
                data = new RenderFlowData(windowWidth, windowHeight);
            }
        }
        if (extra != null) {
            data.extraData.putAll(extra);
        }
        return data;
    }


}

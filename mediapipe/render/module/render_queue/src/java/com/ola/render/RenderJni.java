/**
 * @ProjectName: MyLrcRender
 * @Package: com.ola.render
 * @ClassName: RenderJni
 * @Description:
 * @Author: 王强
 * @CreateDate: 2022/7/19 14:41
 */

package com.ola.render;

public class RenderJni{
    
    static {
        System.loadLibrary("ola_render_jni");
    }

    public static native long create();

    public static native int render(
            long renderContext,
            int textureId,
            int width,
            int height,
            long timestamp,
            boolean export
     );

    public static native void release(long render);
}

/**
 * @ProjectName: MyLrcRender
 * @Package: com.ola.render
 * @ClassName: RenderJni2
 * @Description:
 * @Author: 王强
 * @CreateDate: 2022/7/19 14:41
 */

package com.ola.render;

public class RenderJni{
    static {
        System.loadLibrary("render");
    }

    public static native Long create();

    public static native int render(
            Long renderContext,
            int textureId,
            int width,
            int height,
            Long timestamp,
            boolean export
     );

    public static native void release(Long render);
}

//
//  OlaRender.hpp
//  OlaRender
//
//  Created by 王韧竹 on 2022/6/17.
//

#ifndef OlaRender_hpp
#define OlaRender_hpp

#define USE_OLARENDER 0
// 是否开启多上下文渲染
#define USE_MULTICONTEXT 0
// 是否直接覆盖srcTexture
#define USE_TEXImage2D 0
// 是否要直接渲染到srcTexture上
#define USE_RENDER_TO_SRCTEXTURE 0
// 结束后是否需要还原fbo
#define USE_RESTORE_FBO 0
// 每次渲染结束之后重新创建
#define USE_NEED_RECREATE 0

#include <stdio.h>
#include "macros.h"

#if USE_OLARENDER
#include <target/TargetView.h>
#if PLATFORM == PLATFORM_IOS
#include "OLARenderView.h"
#endif
#endif

#if PLATFORM == PLATFORM_IOS
#import <OpenGLES/ES3/gl.h>
#import <OpenGLES/ES3/glext.h>
#elif PLATFORM == PLATFORM_ANDROID
#include <GLES3/gl3.h>
#include <GLES3/gl3ext.h>

#endif


NS_OLA_BEGIN

    typedef struct {
        int width;
        int height;
        int textureId;
        int ioSurfaceId; // iOS 专属
        int64_t frameTime;
    } TextureInfo;

    class Source;

    class OlaRender {

    public:
        ~OlaRender();

        OlaRender();

#if USE_OLARENDER
        // Android
    static OlaRender* create(void *env, void *context);
#endif

        static OlaRender *create();


        /// 加载测试图  初次使用或者release后需要重新Load
        virtual int loadGraph() = 0;

        virtual int release() = 0;

        virtual TextureInfo render(TextureInfo inputTexture, bool exportFlag) = 0;

#if USE_OLARENDER

#if PLATFORM == PLATFORM_IOS
        virtual void setDisplayView(OLARenderView *target) = 0;
#else
        virtual void setDisplayView(TargetView *target) = 0;
#endif

        virtual void removeRenderTarget() = 0;

        virtual Source* getTerminalSource() = 0;
#endif
    };
NS_OLA_END

#endif /* OlaRender_hpp */

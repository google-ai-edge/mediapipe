//
//  OlaRenderIMP.hpp
//  OlaRender
//
//  Created by 王韧竹 on 2022/6/20.
//

#ifndef OlaRenderIMP_hpp
#define OlaRenderIMP_hpp

#include <stdio.h>
#include "macros.h"
#include "OlaRender.h"
#include <math/math_utils.hpp>

#if USE_OLARENDER
#include <Source.hpp>
#include <BrightnessFilter.hpp>
#include <SobelEdgeDetectionFilter.hpp>
#include <SourceImage.h>
#include <OlaBridgeTextureFilter.hpp>

#if PLATFORM == PLATFORM_IOS
#include <OLARenderView+private.h>

#endif
#else
#if PLATFORM == PLATFORM_IOS
#import <UIKit/UIKit.h>
#endif
#endif

NS_OLA_BEGIN

    class OLARenderIMP : public OlaRender {
    public:
        OLARenderIMP();

        ~OLARenderIMP();

        /// 加载测试图  初次使用或者release后需要重新Load
        virtual int loadGraph() override;

        virtual int release() override;

        virtual TextureInfo render(TextureInfo inputTexture, bool exportFlag) override;

        virtual void setCanvasPixels(int width, int height, const void *pixels,
                                     int64_t frameTime, Vec4 roi);

#if USE_OLARENDER
#if PLATFORM == PLATFORM_IOS
        virtual void setDisplayView(OLARenderView *target) override;

#else
        virtual void setDisplayView(TargetView *target) override;
#endif

        virtual void removeRenderTarget() override;

        virtual Source* getTerminalSource() override;
#endif
    private:
#if USE_OLARENDER
        Framebuffer *_inputFramebuffer = nullptr;
    Filter *_terminalFilter = nullptr;
    BrightnessFilter *_brightFilter = nullptr;
    SobelEdgeDetectionFilter *_sobelFilter = nullptr;
    OlaBridgeTextureFilter *_bridgeFilter = nullptr;
    
#if PLATFORM == PLATFORM_IOS
    OLARenderView *_targetView = nullptr;
#else
    TargetView *_targetView = nullptr;
#endif
#else
        GLuint _outputFramebuffer = -1;
        GLuint _blendProgram = -1;
        GLuint _positionSlot;
        GLuint _positionSlot1;
        GLuint _texCoordSlot;
        GLuint _texCoordSlot1;
        GLuint _inputTextureSlot;
        GLuint _inputTextureSlot1;
        GLuint _outputTexture = -1;
        GLuint _blendTexture = -1;
        GLuint _blendFbo = -1;
        GLuint _blend_mvp = -1;

        GLuint _transformProgram = -1;
        GLuint _transformPositionSlot = -1;
        GLuint _transformTexCoordSlot = -1;
        GLuint _transformTextureSlot = -1;
        GLuint _transform_mvp = -1;
        GLuint _transformTexture = -1;
        Vec2 _lastTransformSize = Vec2(0.0, 0.0);
        int _lastWidth = 0;
        int _lastHeight = 0;
        Mat4 _mvp_matrix;
        Vec4 _roi;

        void _loadProgram();

        GLuint _loadShader(GLenum shaderType, const std::string &shaderString);

        void _setROI(Vec4 roi);

        void _loadOutputTexture(int width, int height);

#if PLATFORM == PLATFORM_IOS
#endif
#endif

        bool _isInit = false;
        float _tempFactor = 0.0;
        int _renderWidth = 0, _renderHeight = 0;


    };

NS_OLA_END

#endif /* OlaRenderIMP_hpp */

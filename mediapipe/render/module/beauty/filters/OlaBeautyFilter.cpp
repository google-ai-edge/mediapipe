#include "OlaBeautyFilter.hpp"
#include "mediapipe/render/core/math/vec2.hpp"

namespace Opipe {
    OlaBeautyFilter::OlaBeautyFilter(Context *context) : FilterGroup(context)
    {

    }

    OlaBeautyFilter::~OlaBeautyFilter()
    {
        if (_lutImage) {
            _lutImage->release();
            _lutImage = nullptr;
        }

        if (_bilateralFilter) {
            _bilateralFilter->release();
            _bilateralFilter = nullptr;
        }

        if (_unSharpMaskFilter) {
            _unSharpMaskFilter->release();
            _unSharpMaskFilter = nullptr;
        }

        if (_alphaBlendFilter) {
            _alphaBlendFilter->release();
            _alphaBlendFilter = nullptr;
        }

        if (_lutFilter) {
            _lutFilter->release();
            _lutFilter = nullptr;
        }

        if (_bilateralAdjustFilter) {
            _bilateralAdjustFilter->release();
            _bilateralAdjustFilter = nullptr;
        }

        if (_faceDistortFilter) {
            _faceDistortFilter->release();
            _faceDistortFilter = nullptr;
        }

        if (_lookUpGroupFilter) {
            _lookUpGroupFilter->release();
            _lookUpGroupFilter = nullptr;
        }
    }

    OlaBeautyFilter *OlaBeautyFilter::create(Context *context)
    {
        OlaBeautyFilter *ret = new (std::nothrow)OlaBeautyFilter(context);
        if (ret && ret->init(context)) {
            return ret;
        } else {
            delete ret;
            return nullptr;
        }
    }

    bool OlaBeautyFilter::init(Context *context) {
        if (!FilterGroup::init(context)) {
            return false;
        }
        _lutFilter = LUTFilter::create(context);
        _unSharpMaskFilter = UnSharpMaskFilter::create(context);
        _unSharpMaskFilter->addTarget(_lutFilter, 0);
        _faceDistortFilter = FaceDistortionFilter::create(context);
        _bilateralAdjustFilter = BilateralAdjustFilter::create(context);
        addFilter(_bilateralAdjustFilter);

        _lookUpGroupFilter = FilterGroup::create(context);
        _lookUpGroupFilter->addFilter(_unSharpMaskFilter);
        
        _alphaBlendFilter = AlphaBlendFilter::create(context);

        _bilateralFilter = BilateralFilter::create(context);
        addFilter(_bilateralFilter);
        
        _bilateralAdjustFilter->addTarget(_lookUpGroupFilter)->addTarget(_alphaBlendFilter, 1);
        
        _bilateralFilter->addTarget(_bilateralAdjustFilter, 1)->addTarget(_alphaBlendFilter, 0);
        
        _alphaBlendFilter->setMix(0.8);
        
        
        _bilateralAdjustFilter->setOpacityLimit(0.6);
        _bilateralFilter->setDistanceNormalizationFactor(2.746);
        _bilateralFilter->setTexelSpacingMultiplier(2.7);
        _unSharpMaskFilter->setBlurRadiusInPixel(4.0f, true);
        _unSharpMaskFilter->setBlurRadiusInPixel(2.0f, false);
        _unSharpMaskFilter->setIntensity(1.365);
        
        _alphaBlendFilter->addTarget(_faceDistortFilter);

        setTerminalFilter(_faceDistortFilter);
        std::vector<Vec2> defaultFace;
        
        registerProperty("face", defaultFace, "人脸点", [this](std::vector<Vec2> facePoints) {
            _faceDistortFilter->setFacePoints(facePoints);
        });

        registerProperty("eye", 0.0f, "大眼 0.0 - 1.0",
                         [this](float eye) {
            _faceDistortFilter->setEye(eye);
        });

        registerProperty("slim", 0.0f, "瘦脸 0.0 - 1.0",
                         [this](float slim) {
            _faceDistortFilter->setSlim(slim);
        });
        
        registerProperty("nose", 0.0f, "瘦鼻 0.0 - 1.0",
                         [this](float nose) {
            _faceDistortFilter->setNose(nose);
        });

        registerProperty("skin", 0.0f, "磨皮 0.0 - 1.0",
                         [this](float skin) {
            if (skin == 0.0) {
                _bilateralAdjustFilter->setEnable(false);
            } else {
                _bilateralAdjustFilter->setEnable(true);
                _bilateralAdjustFilter->setOpacityLimit(skin);
            }
        });

        registerProperty("whiten", 0.0f, "美白 0.0 - 1.0",
                         [this](float whiten) {
            _alphaBlendFilter->setMix(whiten);
        });
        
        
        return true;
        
    }

    bool OlaBeautyFilter::proceed(float frameTime, bool bUpdateTargets) {
        return FilterGroup::proceed(frameTime, bUpdateTargets);
    }

    void OlaBeautyFilter::update(float frameTime) {
        FilterGroup::update(frameTime);
    }

    void OlaBeautyFilter::setLUTImage(SourceImage *lutImage) {
        _lutImage = lutImage;
        _lutImage->retain();
        if (_lutFilter) {
            auto *framebuffer = lutImage->getFramebuffer();
            _lutFilter->setInputFramebuffer(framebuffer, NoRotation, 1, true);
        }
    }


    void OlaBeautyFilter::setInputFramebuffer(Framebuffer *framebuffer,
                                                        RotationMode rotationMode,
                                                        int texIdx,
                                                        bool ignoreForPrepared) {
        for (auto& filter : _filters) {
            filter->setInputFramebuffer(framebuffer, rotationMode, texIdx, ignoreForPrepared);
        }
    }

}

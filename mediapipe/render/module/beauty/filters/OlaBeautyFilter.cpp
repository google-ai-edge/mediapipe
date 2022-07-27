#include "OlaBeautyFilter.hpp"

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
        
        _bilateralFilter = BilateralFilter::create(context);
        addFilter(_bilateralFilter);

        _bilateralAdjustFilter = BilateralAdjustFilter::create(context);
        addFilter(_bilateralAdjustFilter);
        
        _unSharpMaskFilter = UnSharpMaskFilter::create(context);
        addFilter(_unSharpMaskFilter);

        _lutFilter = LUTFilter::create(context);
        _unSharpMaskFilter->addTarget(_lutFilter, 0);

        _lookUpGroupFilter = FilterGroup::create(context);
        _lookUpGroupFilter->addFilter(_unSharpMaskFilter);

        _alphaBlendFilter = AlphaBlendFilter::create(context);
        _faceDistortFilter = FaceDistortionFilter::create(context);
        

        _bilateralFilter->addTarget(_bilateralAdjustFilter, 1)->
        addTarget(_alphaBlendFilter, 0);

        _bilateralAdjustFilter->addTarget(_lookUpGroupFilter)->
        addTarget(_alphaBlendFilter, 1)->addTarget(_faceDistortFilter);
 
        _alphaBlendFilter->setMix(0.0);

        _unSharpMaskFilter->setBlurRadiusInPixel(4.0f, true);
        _unSharpMaskFilter->setBlurRadiusInPixel(2.0f, false);
        _unSharpMaskFilter->setIntensity(1.365);
        
        _bilateralAdjustFilter->setOpacityLimit(0.6);
        
        _bilateralFilter->setDistanceNormalizationFactor(2.746);
        _bilateralFilter->setTexelSpacingMultiplier(2.7);

        setTerminalFilter(_faceDistortFilter);

        std::vector<Vec2> defaultFace;

        
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
        if (_lutFilter) {
            auto *framebuffer = _lutFilter->getFramebuffer();
            framebuffer->resetRetainCount();
            _lutImage->retain();
            _lutImage->addTarget(_lutFilter, 1, true);
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

    void OlaBeautyFilter::setSmoothing(float smoothing) {
        smoothing = smoothing < -1 ? -1 : smoothing;
        smoothing = smoothing > 1 ? 1 : smoothing;
        _bilateralAdjustFilter->setOpacityLimit(smoothing);
    }
    
    float OlaBeautyFilter::getSmoothing() {
        return _bilateralAdjustFilter->getOpacityLimit();
    }
    
    void OlaBeautyFilter::setWhitening(float whitening) {
        _alphaBlendFilter->setMix(whitening);
    }
    
    float OlaBeautyFilter::getWhitening() {
        return _alphaBlendFilter->getMix();
    }

}

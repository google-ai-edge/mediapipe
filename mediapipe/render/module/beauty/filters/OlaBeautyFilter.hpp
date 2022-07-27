#include "mediapipe/render/core/Filter.hpp"
#include "mediapipe/render/core/FilterGroup.hpp"
#include "mediapipe/render/core/BilateralFilter.hpp"
#include "mediapipe/render/core/AlphaBlendFilter.hpp"
#include "mediapipe/render/core/LUTFilter.hpp"
#include "mediapipe/render/core/SourceImage.hpp"
#include "BilateralAdjustFilter.hpp"
#include "UnSharpMaskFilter.hpp"
#include "FaceDistortionFilter.hpp"

namespace Opipe
{
    class OlaBeautyFilter : public FilterGroup
    {
    public:
        float getSmoothing();

        float getWhitening();

        void setSmoothing(float smoothing);

        void setWhitening(float whitening);

    public:
        static OlaBeautyFilter *create(Context *context);

        bool init(Context *context);

        bool proceed(float frameTime = 0, bool bUpdateTargets = true) override;

        void update(float frameTime = 0) override;

        virtual void setInputFramebuffer(Framebuffer *framebuffer,
                                         RotationMode rotationMode =
                                             RotationMode::NoRotation,
                                         int texIdx = 0,
                                         bool ignoreForPrepared = false) override;

        void setLUTImage(SourceImage *image);

        OlaBeautyFilter(Context *context);

        virtual ~OlaBeautyFilter();

        void setFacePoints(std::vector<Vec2> facePoints) {
            _faceDistortFilter->setFacePoints(facePoints);
        }

        // "大眼 0.0 - 1.0"
        void setEye(float eye) {
            _faceDistortFilter->setEye(eye);
        }

        //1.0f, "瘦脸 0.0 - 1.0",
        void setSlim(float slim) {
            _faceDistortFilter->setSlim(slim);
        }

        // "磨皮 0.0 - 1.0"
        void setSkin(float skin) {
            if (skin == 0.0) {
                _bilateralAdjustFilter->setEnable(false);
            } else {
                _bilateralAdjustFilter->setEnable(true);
                _bilateralAdjustFilter->setOpacityLimit(skin);
            }
        }

        // "美白 0.0 - 1.0"
        void setWhiten(float whiten) {
             _alphaBlendFilter->setMix(whiten);
        }

    private:
        BilateralFilter *_bilateralFilter = 0;
        AlphaBlendFilter *_alphaBlendFilter = 0;
        LUTFilter *_lutFilter = 0;
        BilateralAdjustFilter *_bilateralAdjustFilter = 0;
        UnSharpMaskFilter *_unSharpMaskFilter = 0;
        FaceDistortionFilter *_faceDistortFilter = 0;
        FilterGroup *_lookUpGroupFilter = 0;

        SourceImage *_lutImage = 0;
    };
}

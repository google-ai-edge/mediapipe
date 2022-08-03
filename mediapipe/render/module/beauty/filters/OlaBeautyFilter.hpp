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

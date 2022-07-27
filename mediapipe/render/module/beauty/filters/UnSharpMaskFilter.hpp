#ifndef UnSharpMaskFilter_hpp
#define UnSharpMaskFilter_hpp

#include "mediapipe/render/core/FilterGroup.hpp"
#include "mediapipe/render/core/GaussianBlurFilter.hpp"

namespace Opipe {
    class UnSharpMaskFilter : public FilterGroup {
    public:
        void setIntensity(float intensity);
        void setBlurRadiusInPixel(float blurRadius, bool isVertical);
        
    public:
        static UnSharpMaskFilter* create(Context *context);
        
        bool init(Context *context);
    public:
        UnSharpMaskFilter(Context *context);
        ~UnSharpMaskFilter();
        
        GaussianBlurFilter *_blurFilter = nullptr;
        Filter *_unsharpMaskFilter = nullptr;
        
    };
}

#endif
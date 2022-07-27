#ifndef BilateralAdjustFilter_hpp
#define BilateralAdjustFilter_hpp

#include "mediapipe/render/core/Filter.hpp"
#include "mediapipe/render/core/Context.hpp"

namespace Opipe
{
    class BilateralAdjustFilter : public Opipe::Filter
    {
    public:
        static BilateralAdjustFilter *create(Opipe::Context *context);
        bool init(Opipe::Context *context);

        virtual bool proceed(float frameTime = 0, bool bUpdateTargets = true) override;
        float getOpacityLimit() { return _opacityLimit; };
        void setOpacityLimit(float opacityLimit)
        {
            _opacityLimit = opacityLimit;
        }

    public:
        BilateralAdjustFilter(Opipe::Context *context);
        ~BilateralAdjustFilter(){};

        float _opacityLimit;
    };
}

#endif
#ifndef LookUpFilter_hpp
#define LookUpFilter_hpp

#include "Filter.hpp"
#include "Context.hpp"

namespace Opipe
{

    class LUTFilter : public Filter
    {
    public:
        static LUTFilter *create(Context *context);
        bool init(Context *context);
        void setStep(float step);
        virtual bool proceed(float frameTime = 0, bool bUpdateTargets = true) override;
        virtual void update(float frameTime) override;

    public:
        LUTFilter(Context *context);
        ~LUTFilter()
        {
            delete _framebuffer;
            _framebuffer = nullptr;
        };
        float _step;
    };

}

#endif

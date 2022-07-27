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

    public:
        LUTFilter(Context *context);
        ~LUTFilter(){};
        float _step;
    };

}

#endif
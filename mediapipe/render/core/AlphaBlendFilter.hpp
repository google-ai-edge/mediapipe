#ifndef AlphaBlendFilter_cpp
#define AlphaBlendFilter_cpp

#include <stdio.h>
#include "Filter.hpp"

namespace Opipe {
    class AlphaBlendFilter : public virtual Filter {
    public:
        static AlphaBlendFilter* create(Context *context);
        bool init(Context *context);
        
        virtual bool proceed(float fraAlpmeTime = 0.0,
                             bool bUpdateTargets = true) override;
        
        float getMix() {
            return _mix;
        };
        
        void setMix(float mix) {
            _mix = mix;
        }
        
        void setInputFramebuffer(Framebuffer* framebuffer,
                                 RotationMode rotationMode,
                                 int texIdx, bool ignoreForPrepared) override;
        
    public:
        AlphaBlendFilter(Context *context);
        virtual ~AlphaBlendFilter() {};
        float _mix;
        
    };
}

#endif

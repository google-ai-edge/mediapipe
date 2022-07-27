#ifndef BilateralFilter_hpp
#define BilateralFilter_hpp

#include "FilterGroup.hpp"
#include "GPUImageMacros.h"

NS_GI_BEGIN

class BilateralMonoFilter : public Filter {
public:
    enum Type {HORIZONTAL, VERTICAL};
    
    static BilateralMonoFilter* create(Context *context, Type type = HORIZONTAL);
    bool init(Context *context);
    
    virtual bool proceed(float frameTime = 0, bool bUpdateTargets = true) override;
    
    void setTexelSpacingMultiplier(float multiplier);
    void setDistanceNormalizationFactor(float value);
protected:
    BilateralMonoFilter(Context *context, Type type);
    Type _type;
    float _texelSpacingMultiplier;
    float _distanceNormalizationFactor;
};

class BilateralFilter : public FilterGroup {
public:
    virtual ~BilateralFilter();
    
    static BilateralFilter* create(Context *context);
    bool init(Context *context);
    
    void setTexelSpacingMultiplier(float multiplier);
    void setDistanceNormalizationFactor(float value);
    
public:
    BilateralFilter(Context *context);
    
private:
    //friend BilateralMonoFilter;
    BilateralMonoFilter* _hBlurFilter;
    BilateralMonoFilter* _vBlurFilter;
};


NS_GI_END


#endif
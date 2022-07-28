#include "AlphaBlendFilter.hpp"

namespace Opipe {
    const std::string kAlphaBlendFragmentShaderString = SHADER_STRING
    (
         varying highp vec2 vTexCoord;
         varying highp vec2 vTexCoord1;
         uniform sampler2D colorMap;
         uniform sampler2D colorMap1;
         uniform lowp float mixturePercent;
         void main() {
             lowp vec4 textureColor = texture2D(colorMap, vTexCoord);
             lowp vec4 textureColor2 = texture2D(colorMap1, vTexCoord1);
             gl_FragColor = vec4(mix(textureColor.rgb, textureColor2.rgb, textureColor2.a * mixturePercent), textureColor.a);
         }
     );
    AlphaBlendFilter::AlphaBlendFilter(Context *context) : Filter(context), _mix(1.0) {
        
    }
    
    AlphaBlendFilter* AlphaBlendFilter::create(Context *context) {
        AlphaBlendFilter* ret = new (std::nothrow) AlphaBlendFilter(context);
        if (!ret || !ret->init(context)) {
            delete ret;
            ret = 0;
        }
        return ret;
    }
    
    bool AlphaBlendFilter::init(Context *context) {
        if (!Filter::initWithFragmentShaderString(context,
                                                  kAlphaBlendFragmentShaderString,
                                                  2)) {
            return false;
        }
        return true;
    }

    void AlphaBlendFilter::setInputFramebuffer(Framebuffer* framebuffer,
                                               RotationMode rotationMode,
                                               int texIdx, bool ignoreForPrepared) {
        Filter::setInputFramebuffer(framebuffer, rotationMode, texIdx, ignoreForPrepared);
    }

    bool AlphaBlendFilter::proceed(float frameTime,
                                          bool bUpdateTargets/* = true*/) {
        _filterProgram->setUniformValue("mixturePercent", _mix);
        return Filter::proceed(frameTime, bUpdateTargets);
    }
}

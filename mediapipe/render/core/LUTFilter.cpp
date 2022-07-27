#include "LUTFilter.hpp"

namespace Opipe
{
    const std::string kLookupFragmentShaderString = SHADER_STRING(
        varying highp vec2 vTexCoord;
        varying highp vec2 vTexCoord1; // TODO: This is not used

        uniform sampler2D colorMap;
        uniform sampler2D colorMap1; // lookup texture
        uniform lowp float step;

        void main() {
            highp vec4 textureColor = texture2D(colorMap, vTexCoord);

            highp float blueColor = textureColor.b * 63.0;

            highp vec2 quad1;
            quad1.y = floor(floor(blueColor) / 8.0);
            quad1.x = floor(blueColor) - (quad1.y * 8.0);

            highp vec2 quad2;
            quad2.y = floor(ceil(blueColor) / 8.0);
            quad2.x = ceil(blueColor) - (quad2.y * 8.0);

            highp vec2 texPos1;
            texPos1.x = (quad1.x * 0.125) + 0.5 / 512.0 + ((0.125 - 1.0 / 512.0) * textureColor.r);
            texPos1.y = (quad1.y * 0.125) + 0.5 / 512.0 + ((0.125 - 1.0 / 512.0) * textureColor.g);

            highp vec2 texPos2;
            texPos2.x = (quad2.x * 0.125) + 0.5 / 512.0 + ((0.125 - 1.0 / 512.0) * textureColor.r);
            texPos2.y = (quad2.y * 0.125) + 0.5 / 512.0 + ((0.125 - 1.0 / 512.0) * textureColor.g);

            lowp vec4 newColor1 = texture2D(colorMap1, texPos1);
            lowp vec4 newColor2 = texture2D(colorMap1, texPos2);

            lowp vec4 newColor = mix(newColor1, newColor2, fract(blueColor));
            lowp vec3 finalColor = mix(textureColor.rgb, newColor.rgb, step);

            gl_FragColor = vec4(finalColor, textureColor.w);
        });

    LUTFilter::LUTFilter(Opipe::Context *context) : Filter(context), _step(1.0)
    {
    }

    Opipe::LUTFilter *LUTFilter::create(Opipe::Context *context)
    {
        LUTFilter *ret = new (std::nothrow) LUTFilter(context);
        if (!ret || !ret->init(context))
        {
            delete ret;
            ret = 0;
        }
        return ret;
    }

    bool LUTFilter::init(Opipe::Context *context)
    {
        if (!Opipe::Filter::initWithFragmentShaderString(context, kLookupFragmentShaderString, 2))
        {
            return false;
        }
        return true;
    }

    void LUTFilter::setStep(float step)
    {
        _step = step;
    }

    bool LUTFilter::proceed(float frameTime, bool bUpdateTargets /* = true*/)
    {
        _filterProgram->setUniformValue("step", _step);
        return Filter::proceed(frameTime, bUpdateTargets);
    }
}

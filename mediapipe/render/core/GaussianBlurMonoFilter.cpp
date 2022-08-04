/*
 * GPUImage-x
 *
 * Copyright (C) 2017 Yijin Wang, Yiqian Wang
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cmath>
#include "GaussianBlurMonoFilter.hpp"

NS_GI_BEGIN

REGISTER_FILTER_CLASS(GaussianBlurMonoFilter)

GaussianBlurMonoFilter::GaussianBlurMonoFilter(Context *context, Type type/* = HORIZONTAL*/) : Filter(context)
,_type(type)
,_radius(4)
,_sigma(2.0)
,_multiplier(1.0)
{
}

GaussianBlurMonoFilter* GaussianBlurMonoFilter::create(Context *context, Type type/* = HORIZONTAL*/, int radius/* = 4*/, float sigma/* = 2.0*/, float multiplier) {
    GaussianBlurMonoFilter* ret = new (std::nothrow) GaussianBlurMonoFilter(context, type);
    if (ret && !ret->init(context, radius, sigma, multiplier)) {
        delete ret;
        ret = 0;
    }
    return ret;
}

bool GaussianBlurMonoFilter::init(Context *context, int radius, float sigma, float multiplier) {
    if (Filter::initWithShaderString(context, _generateOptimizedVertexShaderString(radius, sigma), _generateOptimizedFragmentShaderString(radius, sigma))) {
        return true;
    }
    return false;
}

void GaussianBlurMonoFilter::setRadius(int radius) {
    if (radius == _radius) return;
    
    _radius = radius;
    
    if (_filterProgram) {
        delete _filterProgram;
        _filterProgram = 0;
    }
    initWithShaderString(_context, _generateOptimizedVertexShaderString(_radius, _sigma), _generateOptimizedFragmentShaderString(_radius, _sigma));
}

void GaussianBlurMonoFilter::setSigma(float sigma) {
    if (sigma == _sigma) return;
    
    _sigma = round(sigma);
    
    int calculatedSampleRadius = 0;
    if (_sigma >= 1) // Avoid a divide-by-zero error here
    {
        // Calculate the number of pixels to sample from by setting a bottom limit for the contribution of the outermost pixel
        float minimumWeightToFindEdgeOfSamplingArea = 1.0/256.0;
        calculatedSampleRadius = floor(sqrt(-2.0 * pow(_sigma, 2.0) * log(minimumWeightToFindEdgeOfSamplingArea * sqrt(2.0 * M_PI * pow(_sigma, 2.0))) ));
        calculatedSampleRadius += calculatedSampleRadius % 2; // There's nothing to gain from handling odd radius sizes, due to the optimizations I use
    }
    _radius = calculatedSampleRadius;
    
    if (_filterProgram) {
        delete _filterProgram;
        _filterProgram = 0;
    }
    initWithShaderString(_context, _generateOptimizedVertexShaderString(_radius, _sigma), _generateOptimizedFragmentShaderString(_radius, _sigma));
}

bool GaussianBlurMonoFilter::proceed(float frameTime, bool bUpdateTargets/* = true*/) {

    RotationMode inputRotation = _inputFramebuffers.begin()->second.rotationMode;

    if (rotationSwapsSize(inputRotation))
    {
        if (_type == HORIZONTAL) {
            _filterProgram->setUniformValue("texelWidthOffset", (float)0.0);
            _filterProgram->setUniformValue("texelHeightOffset", (float)(_multiplier / _framebuffer->getWidth()));
        } else {
            _filterProgram->setUniformValue("texelWidthOffset", (float)(_multiplier / _framebuffer->getHeight()));
            _filterProgram->setUniformValue("texelHeightOffset", (float)0.0);
        }
    } else {
        if (_type == HORIZONTAL) {
            _filterProgram->setUniformValue("texelWidthOffset", (float)(_multiplier / _framebuffer->getWidth()));
            _filterProgram->setUniformValue("texelHeightOffset", (float)0.0);
        } else {
            _filterProgram->setUniformValue("texelWidthOffset", (float)0.0);
            _filterProgram->setUniformValue("texelHeightOffset", (float)(_multiplier / _framebuffer->getHeight()));
        }
    }
    return Filter::proceed(frameTime, bUpdateTargets);
}

std::string GaussianBlurMonoFilter::_generateVertexShaderString(int radius, float sigma) {
    if (radius < 1 || sigma <= 0.0)
    {
        return kDefaultVertexShader;
    }

    std::string shaderStr =
            str_format("\
        attribute vec4 position;\n\
        attribute vec4 texCoord;\n\
        uniform float texelWidthOffset;\n\
        uniform float texelHeightOffset;\n\
        varying vec2 blurCoordinates[%d];\n\
        void main()\n\
        {\n\
            gl_Position = position;\n\
            vec2 texelSpacing = vec2(texelWidthOffset, texelHeightOffset);\n\
        ", radius * 2 + 1);

    for (int i = 0; i < radius * 2 + 1; ++i) {
        int offsetFromCenter = i - radius;
        if (offsetFromCenter == 0) {
            shaderStr = shaderStr + str_format("blurCoordinates[%d] = texCoord.xy;\n", i);
        } else {
            shaderStr = shaderStr + str_format("blurCoordinates[%d] = texCoord.xy + texelSpacing * (%f);\n", i, (float)offsetFromCenter);
        }
    }

    shaderStr += "}\n";

    return shaderStr;
}

std::string GaussianBlurMonoFilter::_generateFragmentShaderString(int radius, float sigma) {
    if (radius < 1 || sigma <= 0.0)
    {
        return kDefaultFragmentShader;
    }

    float* standardGaussianWeights = new float[radius + 1];
    float sumOfWeights = 0.0;
    for (int i = 0; i < radius + 1; ++i)
    {
        standardGaussianWeights[i] = (1.0 / sqrt(2.0 * PI * pow(sigma, 2.0))) * exp(-pow(i, 2.0) / (2.0 * pow(sigma, 2.0)));

        if (i == 0)
        {
            sumOfWeights += standardGaussianWeights[i];
        } else {
            sumOfWeights += 2.0 * standardGaussianWeights[i];
        }
    }
    for (int i = 0; i < radius + 1; ++i)
    {
        standardGaussianWeights[i] = standardGaussianWeights[i] / sumOfWeights;
    }

    std::string shaderStr =
            str_format("\
           uniform sampler2D colorMap;\n\
           varying highp vec2 blurCoordinates[%d];\n\
           void main()\n\
           {\n\
               gl_FragColor = vec4(0.0);\n", radius * 2 + 1);
    for (int i = 0; i < radius * 2 + 1; ++i) {
        int offsetFromCenter = i - radius;
        shaderStr += str_format("gl_FragColor += texture2D(colorMap, blurCoordinates[%d]) * %f;\n", i, standardGaussianWeights[abs(offsetFromCenter)]);
    }
    shaderStr += "}";

    delete[] standardGaussianWeights; standardGaussianWeights = 0;

    return shaderStr;
}

std::string GaussianBlurMonoFilter::_generateOptimizedVertexShaderString(int radius, float sigma)
{
    if (radius < 1 || sigma <= 0.0)
    {
        return kDefaultVertexShader;
    }
    
    // 1. generate the normal Gaussian weights for a given sigma
    float* standardGaussianWeights = new float[radius + 1];
    float sumOfWeights = 0.0;
    for (int i = 0; i < radius + 1; ++i)
    {
        standardGaussianWeights[i] = (1.0 / sqrt(2.0 * M_PI * pow(sigma, 2.0))) * exp(-pow(i, 2.0) / (2.0 * pow(sigma, 2.0)));
        if (i == 0)
            sumOfWeights += standardGaussianWeights[i];
        else
            sumOfWeights += 2.0 * standardGaussianWeights[i];
    }
    
    // 2. normalize these weights to prevent the clipping of the Gaussian curve at the end of the discrete samples from reducing luminance
    for (int i = 0; i < radius + 1; ++i)
    {
        standardGaussianWeights[i] = standardGaussianWeights[i] / sumOfWeights;
    }
    
    // 3. From these weights we calculate the offsets to read interpolated values from
    int numberOfOptimizedOffsets = fmin(radius / 2 + (radius % 2), 7);
    float* optimizedGaussianOffsets = new float[numberOfOptimizedOffsets];
    
    for (int i = 0; i < numberOfOptimizedOffsets; ++i)
    {
        GLfloat firstWeight = standardGaussianWeights[i * 2 + 1];
        GLfloat secondWeight = standardGaussianWeights[i * 2 + 2];
        
        GLfloat optimizedWeight = firstWeight + secondWeight;
        
        optimizedGaussianOffsets[i] = (firstWeight * (i * 2 + 1) + secondWeight * (i * 2 + 2)) / optimizedWeight;
    }
    
    std::string shaderStr =
    str_format("\
               attribute vec4 position;\n\
               attribute vec4 texCoord;\n\
               uniform float texelWidthOffset;\n\
               uniform float texelHeightOffset;\n\
               varying highp vec2 blurCoordinates[%d];\n\
               void main()\n\
               {\n\
               gl_Position = position;\n\
               vec2 texelSpacing = vec2(texelWidthOffset, texelHeightOffset);\n\
               ", numberOfOptimizedOffsets * 2 + 1);

    shaderStr = shaderStr + str_format("blurCoordinates[0] = texCoord.xy;\n");
    for (int i = 0; i < numberOfOptimizedOffsets; ++i) {
        shaderStr = shaderStr + str_format(
           "blurCoordinates[%d] = texCoord.xy + texelSpacing * (%f);\n\
            blurCoordinates[%d] = texCoord.xy - texelSpacing * (%f);",
            i * 2 + 1,
            optimizedGaussianOffsets[i],
            i * 2 + 2,
            optimizedGaussianOffsets[i]);
    }
    
    shaderStr += "}\n";
    
    delete[] standardGaussianWeights;
    delete[] optimizedGaussianOffsets;
    
    return shaderStr;
}

std::string GaussianBlurMonoFilter::_generateOptimizedFragmentShaderString(int radius, float sigma)
{
    if (radius < 1 || sigma <= 0.0)
    {
        return kDefaultFragmentShader;
    }
    
    // 1. generate the normal Gaussian weights for a given sigma
    float* standardGaussianWeights = new float[radius + 1];
    float sumOfWeights = 0.0;
    for (int i = 0; i < radius + 1; ++i)
    {
        standardGaussianWeights[i] = (1.0 / sqrt(2.0 * M_PI * pow(sigma, 2.0))) * exp(-pow(i, 2.0) / (2.0 * pow(sigma, 2.0)));
        if (i == 0)
            sumOfWeights += standardGaussianWeights[i];
        else
            sumOfWeights += 2.0 * standardGaussianWeights[i];
    }
    
    // 2. normalize these weights to prevent the clipping of the Gaussian curve at the end of the discrete samples from reducing luminance
    for (int i = 0; i < radius + 1; ++i)
    {
        standardGaussianWeights[i] = standardGaussianWeights[i] / sumOfWeights;
    }
    
    // 3. From these weights we calculate the offsets to read interpolated values from
    int trueNumberOfOptimizedOffsets = radius / 2 + (radius % 2);
    int numberOfOptimizedOffsets = fmin(trueNumberOfOptimizedOffsets, 7);

    std::string shaderStr =
    str_format("\
               uniform sampler2D colorMap;\n\
               uniform highp float texelWidthOffset;\n\
               uniform highp float texelHeightOffset;\n\
               varying highp vec2 blurCoordinates[%d];\n\
               void main()\n\
               {\n\
               gl_FragColor = vec4(0.0);\n", numberOfOptimizedOffsets * 2 + 1);
    
    shaderStr += str_format("gl_FragColor += texture2D(colorMap, blurCoordinates[0]) * %f;\n", standardGaussianWeights[0]);
    for (int i = 0; i < numberOfOptimizedOffsets; ++i) {
        float firstWeight = standardGaussianWeights[i * 2 + 1];
        float secondWeight = standardGaussianWeights[i * 2 + 2];
        float optimizedWeight = firstWeight + secondWeight;
        
        shaderStr += str_format("gl_FragColor += texture2D(colorMap, blurCoordinates[%d]) * %f;\n", i * 2 + 1, optimizedWeight);
        shaderStr += str_format("gl_FragColor += texture2D(colorMap, blurCoordinates[%d]) * %f;\n", i * 2 + 2, optimizedWeight);
    }
    
    // If the number of required samples exceeds the amount we can pass in via varyings, we have to do dependent texture reads in the fragment shader
    if (trueNumberOfOptimizedOffsets > numberOfOptimizedOffsets)
    {
        shaderStr += str_format("highp vec2 texelSpacing = vec2(texelWidthOffset, texelHeightOffset);\n");
        
        for (int i = numberOfOptimizedOffsets; i < trueNumberOfOptimizedOffsets; i++)
        {
            float firstWeight = standardGaussianWeights[i * 2 + 1];
            float secondWeight = standardGaussianWeights[i * 2 + 2];
            
            float optimizedWeight = firstWeight + secondWeight;
            float optimizedOffset = (firstWeight * (i * 2 + 1) + secondWeight * (i * 2 + 2)) / optimizedWeight;
            
            shaderStr += str_format("gl_FragColor += texture2D(colorMap, blurCoordinates[0] + texelSpacing * %f) * %f;\n", optimizedOffset, optimizedWeight);
            
            shaderStr += str_format("gl_FragColor += texture2D(colorMap, blurCoordinates[0] - texelSpacing * %f) * %f;\n", optimizedOffset, optimizedWeight);
        }
    }

    shaderStr += "}";
    
    delete[] standardGaussianWeights; standardGaussianWeights = 0;
    return shaderStr;
}

NS_GI_END

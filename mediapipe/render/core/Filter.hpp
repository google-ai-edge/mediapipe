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

#ifndef Filter_hpp
#define Filter_hpp
#include "string"
#include "GPUImageMacros.h"
#include "Source.hpp"
#include "Target.hpp"
#include "GLProgram.hpp"
#include "Ref.hpp"
#include "GPUImageUtil.h"


NS_GI_BEGIN


// Hardcode the vertex shader for standard filters, but this can be overridden
const std::string kDefaultVertexShader = SHADER_STRING
(
 attribute vec4 position;
 attribute vec4 texCoord;
 
 // uniform mat4 mvp;
 
 varying vec2 vTexCoord;
 
 
 void main()
 {
    //     gl_Position = mvp * position;
    gl_Position = position;
    vTexCoord = texCoord.xy;
}
 );

const std::string kDefaultFragmentShader = SHADER_STRING
(
 varying highp vec2 vTexCoord;
 uniform sampler2D colorMap;

 void main()
 {
     gl_FragColor = texture2D(colorMap, vTexCoord);
 }
 );

const std::string kDefaultDisplayFragmentShader = SHADER_STRING
(
 precision highp float;
 varying highp vec2 vTexCoord;
 uniform sampler2D colorMap;
 void main()
 {
     vec4 color = texture2D(colorMap, vTexCoord);
     gl_FragColor = vec4(color.rgb, 1);
 }
 );

const std::string kDefaultCaptureFragmentShader = SHADER_STRING
(
 precision highp float;
 varying highp vec2 vTexCoord;
 uniform sampler2D colorMap;
 void main()
 {
     vec4 color = texture2D(colorMap, vTexCoord);
     gl_FragColor = vec4(color.rgb, 1);
 }
 );

class Context;
class Filter : public Source, public Target {
public:
    virtual ~Filter();
    
    Filter();
    
    static Filter* create(Context *context, const std::string& filterClassName);
    static Filter* createWithShaderString(Context *context,
                                          const std::string& vertexShaderSource,
                                          const std::string& fragmentShaderSource);
    static Filter* createWithFragmentShaderString(Context *context,
                                                  const std::string& fragmentShaderSource,
                                                  int inputNumber = 1);
    
    bool initWithShaderString(Context *context, const std::string& vertexShaderSource, const std::string& fragmentShaderSource);
    virtual bool initWithFragmentShaderString(Context *context,
                                              const std::string& fragmentShaderSource,
                                              int inputNumber = 1);
    
    void setFilterClassName(const std::string filterClassName) {
        _filterClassName = filterClassName;
    }
    std::string getFilterClassName() const { return _filterClassName; };
    
    virtual void update(float frameTime) override;
    virtual bool proceed(float frameTime = 0.0,
                         bool bUpdateTargets = true) override;
    
    virtual void setInputFramebuffer(Framebuffer* framebuffer,
                             RotationMode rotationMode = NoRotation,
                             int texIdx = 0, bool ignoreForPrepared = false) override;
    GLProgram* getProgram() const { return _filterProgram; };

    // property setters & getters
    virtual bool registerProperty(const std::string& name, int defaultValue, const std::string& comment = "", std::function<void(int&)> setCallback = 0);
    virtual bool registerProperty(const std::string& name, float defaultValue, const std::string& comment = "", std::function<void(float&)> setCallback = 0);
    virtual bool registerProperty(const std::string& name, const std::string& defaultValue, const std::string& comment = "", std::function<void(std::string&)> setCallback = 0);
    bool registerProperty(const std::string& name,
                          std::vector<Vec2> defaultValue,
                          const std::string& comment = "",
                          std::function<void(std::vector<Vec2>&)> setCallback = 0);
    
    bool registerProperty(const std::string& name,
                          Vec2 defaultValue,
                          const std::string& comment = "",
                          std::function<void(Vec2&)> setCallback = 0);
    bool setProperty(const std::string& name, Vec2 value);
    
    bool setProperty(const std::string& name, int value);
    bool setProperty(const std::string& name, float value);
    bool setProperty(const std::string& name, std::string value);
    bool setProperty(const std::string& name, std::vector<Vec2> retValue);
    bool getProperty(const std::string& name, std::vector<Vec2>& retValue);
    bool getProperty(const std::string& name, int& retValue);
    bool getProperty(const std::string& name, float& retValue);
    bool getProperty(const std::string& name, std::string& retValue);
    bool hasProperty(const std::string& name);
    bool hasProperty(const std::string& name, const std::string type);
    bool getPropertyComment(const std::string& name, std::string& retComment);
    bool getPropertyType(const std::string& name, std::string& retType);
    
    bool isEnable() {
        return _enable;
    }
    
    bool isForceEnable() {
        return _forceEnable;
    }
    
    void setEnable(bool enable) {
        if (_forceEnable) {
            //强制设定时外部不可更改
            return;
        }
        _enable = enable;
    }
    
    void setForceEnable(bool force, bool enable) {
        _forceEnable = force;
        _enable = enable;
    }
    
    void setContext(Context *context) {
        _context = context;
    }
#if defined(__ANDROID__) || defined(ANDROID)
    class Registry {
    public:
        Registry(const std::string& name, std::function<Filter*()> createFunc) {
            Filter::_registerFilterClass(name, createFunc);
        }
    };
    static void _registerFilterClass(const std::string& filterClassName, std::function<Filter*()> createFunc) {
        //Log("jin", "Filter：：registerClass : %s", filterClassName.c_str());
        _filterFactories[filterClassName] = createFunc;
    }
#endif

public:
    virtual void filter_externDraw();
    struct {
        float r; float g; float b; float a;
    } _backgroundColor;
    
    // 外部告知需要旋转的方向
    void setTargetRotationMode(RotationMode rotation) {
        _targetRotation = rotation;
    };

    RotationMode getTargetRotationMode() {
        return _targetRotation;
    }
public:
  
    bool useScaleResolution() {
        return _useScaleResolution;
    }
    
    Vector2 getScaleResolution() {
        return _scaleResolution;
    }
    
    void setScaleResolution(Vector2 resolution) {
        _useScaleResolution = true;
        _scaleResolution = resolution;
    }
    
protected:
    Vector4 _roi = Vector4(0.0, 0.0, 1.0, 1.0);
    float _rotate = 0.0;
    
    
protected:
    
    RotationMode _targetRotation = NoRotation;
    GLProgram* _filterProgram = nullptr;
    GLuint _filterPositionAttribute = -1;
    GLint _uniform_mvp = -1;
    
    std::string _filterClassName;
    float _frameCount = 0.0;
    
    Filter(Context *context);
    std::string _getVertexShaderString() const;
    const GLfloat* _getTexureCoordinate(const RotationMode& rotationMode) const;
    
    // properties
    struct Property {
        std::string type;
        std::string comment;
    };
    
    struct Vec2ArrayProperty : Property {
        std::vector<Vec2> value;
        std::function<void(std::vector<Vec2>&)> setCallback;
    };
    std::map<std::string, Vec2ArrayProperty> _vec2ArrayProperties;
    
    struct Vec2Property : Property {
        Vec2 value;
        std::function<void(Vec2&)> setCallback;
    };
    
    virtual Property* _getProperty(const std::string& name);

    std::map<std::string, Vec2Property> _vec2Properties;
    
    struct Vec3Property : Property {
        Vec3 value;
        std::function<void(Vec3&)> setCallback;
    };
    
    struct IntProperty : Property {
        int value;
        std::function<void(int&)> setCallback;
    };
    std::map<std::string, IntProperty> _intProperties;

    struct FloatProperty : Property {
        float value;
        std::function<void(float&)> setCallback;
    };
    std::map<std::string, FloatProperty> _floatProperties;
    
    struct StringProperty : Property {
        std::string value;
        std::function<void(std::string&)> setCallback;
    };
    std::map<std::string, StringProperty> _stringProperties;
    
    Context *getContext();
protected:
    void generateVBOBuffers();
    bool _enable = true;
    bool _forceEnable = false;
    Opipe::Mat4 _mvp_matrix;
    Vector2 _scaleResolution = Vector2(0.0, 0.0);
    bool _useScaleResolution = false;
    
public:
    void releaseVBOBuffers();
private:
    static std::map<std::string, std::function<Filter*()>> _filterFactories;
};

//#if defined(__ANDROID__) || defined(ANDROID)
//#define REGISTER_FILTER_CLASS(className) \
//class className##Registry { \
//    public: \
//        static Filter* newInstance() { \
//            return className::create(); \
//        } \
//    private: \
//        static const Filter::Registry _registry; \
//}; \
//const Filter::Registry className##Registry::_registry(#className, className##Registry::newInstance);
//#elif defined(__APPLE__)
#define REGISTER_FILTER_CLASS(className) 
//#endif

NS_GI_END

#endif /* Filter_hpp */

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


#include "Context.hpp"
#include "Filter.hpp"

NS_GI_BEGIN

std::map<std::string, std::function<Filter*()>> Filter::_filterFactories;
//static GLuint vertexArray = -1;
//static GLuint elementArray[8];

Filter::Filter(Context *context) : Source(context)
,_filterProgram(0)
,_filterClassName("")
{
    _mvp_matrix.set_identity();
    _backgroundColor.r = 1.0;
    _backgroundColor.g = 1.0;
    _backgroundColor.b = 1.0;
    _backgroundColor.a = 1.0;
}

Filter::Filter() {
    
}

Filter::~Filter() {
    if (_filterProgram) {
        delete _filterProgram;
        _filterProgram = 0;
    }
}

void Filter::generateVBOBuffers()
{
    if (_context->vertexArray == -1) {
        GLfloat textureCoordinates[] = {
            //noRotationTextureCoordinates
            -1.0f, -1.0f,   //v0
            0.0f, 0.0f,     //c0
            1.0f, -1.0f,    //v1
            1.0f, 0.0f,     //c1
            -1.0f,  1.0f,   //v2
            0.0f, 1.0f,     //c2
            1.0f,  1.0f,    //v3
            1.0f, 1.0f,     //c3
            
            //rotateLeftTextureCoordinates
            -1.0f, -1.0f,   //v0
            1.0f, 0.0f,     //c0
            1.0f, -1.0f,    //v1
            1.0f, 1.0f,    //c1
            -1.0f,  1.0f,   //v2
            0.0f, 0.0f,    //c2
            1.0f,  1.0f,    //v3
            0.0f, 1.0f,    //c3
            
            //rotateRightTextureCoordinates
            -1.0f, -1.0f,   //v0
            0.0f, 1.0f,     //c0
            1.0f, -1.0f,    //v1
            0.0f, 0.0f,     //c1
            -1.0f,  1.0f,   //v2
            1.0f, 1.0f,     //c2
            1.0f,  1.0f,    //v3
            1.0f, 0.0f,     //c3
            
            //attach
            1.0f, -1.0f,    //v1
            0.0f, 1.0f,     //c1
            -1.0f,  1.0f,   //v2
            1.0f, 0.0f,     //c2
            -1.0f, -1.0f,   //v0
            1.0f, 1.0f,     //c0
            1.0f,  1.0f,    //v3
            0.0f, 0.0f,     //c3
            
        };
        
        CHECK_GL(glGenBuffers(1, &(_context->vertexArray)));
        CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, _context->vertexArray));
        CHECK_GL(glBufferData(GL_ARRAY_BUFFER, 16 * sizeof(GLfloat) * 4, textureCoordinates, GL_STATIC_DRAW));
        
        CHECK_GL(glGenBuffers(8,  _context->elementArray));
        {
            //noRotationTextureCoordinates
            GLushort indices[4] = {0, 1, 2, 3};
            CHECK_GL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,  _context->elementArray[0]));
            CHECK_GL(glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * sizeof(GLushort), indices, GL_STATIC_DRAW));
        }
        
        {
            //rotateLeftTextureCoordinates
            GLushort indices[4] = {4, 5, 6, 7};
            CHECK_GL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,  _context->elementArray[1]));
            CHECK_GL(glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * sizeof(GLushort), indices, GL_STATIC_DRAW));
        }

        {
            //rotateRightTextureCoordinates
            GLushort indices[4] = {8, 9, 10, 11};
            CHECK_GL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,  _context->elementArray[2]));
            CHECK_GL(glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * sizeof(GLushort), indices, GL_STATIC_DRAW));
        }

        {
            //verticalFlipTextureCoordinates
            GLushort indices[4] = {8, 5, 6, 11};
            CHECK_GL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,  _context->elementArray[3]));
            CHECK_GL(glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * sizeof(GLushort), indices, GL_STATIC_DRAW));
        }

        {
            //horizontalFlipTextureCoordinates
            GLushort indices[4] = {4, 9, 10, 7};
            CHECK_GL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,  _context->elementArray[4]));
            CHECK_GL(glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * sizeof(GLushort), indices, GL_STATIC_DRAW));
        }

        {
            //rotateRightVerticalFlipTextureCoordinates
            GLushort indices[4] = {0, 12, 13, 3};
            CHECK_GL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,  _context->elementArray[5]));
            CHECK_GL(glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * sizeof(GLushort), indices, GL_STATIC_DRAW));
        }

        {
            //rotateRightHorizontalFlipTextureCoordinates
            GLushort indices[4] = {14, 1, 2, 15};
            CHECK_GL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,  _context->elementArray[6]));
            CHECK_GL(glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * sizeof(GLushort), indices, GL_STATIC_DRAW));
        }

        {
            //rotate180TextureCoordinates
            GLushort indices[4] = {14, 12, 13, 15};
            CHECK_GL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _context->elementArray[7]));
            CHECK_GL(glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * sizeof(GLushort), indices, GL_STATIC_DRAW));
        }
        
        CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, 0));
        CHECK_GL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
    }
}

//void Filter::releaseVBOBuffers()
//{
//    if (vertexArray != -1) {
//        CHECK_GL(glDeleteBuffers(0, &vertexArray));
//        vertexArray = -1;
//        CHECK_GL(glDeleteBuffers(8, elementArray));
//        for (int i = 0; i < 8; i++) {
//            elementArray[i] = -1;
//        }
//    }
//}

Filter* Filter::create(Context *context, const std::string& filterClassName) {
    std::map<std::string, std::function<Filter*()>>::iterator it = _filterFactories.find(filterClassName);
    if (it == _filterFactories.end())
        return 0;
    else {
        Filter* filter = it->second();
        filter->setFilterClassName(filterClassName);
        return it->second();
    }
}

Filter* Filter::createWithShaderString(Context *context,
                                       const std::string& vertexShaderSource,
                                       const std::string& fragmentShaderSource) {
    Filter* filter = new Filter(context);
    if (!filter->initWithShaderString(context, vertexShaderSource, fragmentShaderSource)) {
        delete filter;
        filter = 0;
        return 0;
    }
    return filter;
}

Filter* Filter::createWithFragmentShaderString(Context *context,
                                               const std::string& fragmentShaderSource,
                                               int inputNumber/* = 1*/) {
    Filter* filter = new Filter(context);
    if (!filter->initWithFragmentShaderString(context, fragmentShaderSource,inputNumber)) {
        delete filter;
        filter = 0;
        return 0;
    }
    return filter;
}

bool Filter::initWithShaderString(Context *context,
                                  const std::string& vertexShaderSource,
                                  const std::string& fragmentShaderSource) {
    
    _filterProgram = GLProgram::createByShaderString(context, vertexShaderSource, fragmentShaderSource);
    if (_filterProgram == NULL) {
        return false; //shader创建失败
    }
//    _context = context;

    _filterPositionAttribute = _filterProgram->getAttribLocation("position");
    _uniform_mvp = _filterProgram->getUniformLocation("mvp");
    getContext()->setActiveShaderProgram(_filterProgram);
    if (_filterPositionAttribute != -1) {
        CHECK_GL(glEnableVertexAttribArray(_filterPositionAttribute));
    }

    return true;
}

bool Filter::initWithFragmentShaderString(Context *context,
                                          const std::string& fragmentShaderSource,
                                          int inputNumber/* = 1*/) {
    _inputNum = inputNumber;
    return initWithShaderString(context, _getVertexShaderString(), fragmentShaderSource);
}

std::string Filter::_getVertexShaderString() const {

    if (_inputNum <= 1)
    {
        return kDefaultVertexShader;
    }
    
    std::string shaderStr = "\
                attribute vec4 position;\n\
                attribute vec4 texCoord;\n\
                varying vec2 vTexCoord;\n\
                ";
    for (int i = 1; i < _inputNum; ++i) {
        shaderStr += str_format("\
                attribute vec4 texCoord%d;\n\
                varying vec2 vTexCoord%d;\n\
                                ", i, i);
    }
    shaderStr += "\
                void main()\n\
                {\n\
                    gl_Position = position;\n\
                    vTexCoord = texCoord.xy;\n\
        ";
    for (int i = 1; i < _inputNum; ++i) {
        shaderStr += str_format("vTexCoord%d = texCoord%d.xy;\n", i, i);
    }
    shaderStr += "}\n";
    
    return shaderStr;
}

void Filter::setInputFramebuffer(Framebuffer* framebuffer,
                                 RotationMode rotationMode,
                                 int texIdx, bool ignoreForPrepared) {
    Target::setInputFramebuffer(framebuffer, rotationMode, texIdx, ignoreForPrepared);
}

bool Filter::proceed(float frameTime, bool bUpdateTargets/* = true*/) {
    if (_framebuffer->isDealloc) {
        return false;
    }
    auto logstr = str_format(" QuarameraLayerGLRender:%s渲染耗时", typeid(*this).name());
    {
#if DEBUG
        _framebuffer->lock(typeid(*this).name());
#else
		_framebuffer->lock();
#endif
        generateVBOBuffers();
        
        getContext()->setActiveShaderProgram(_filterProgram);
        _framebuffer->active();
        
        _filterProgram->setUniformValue("iTime", _frameCount);
        _frameCount += 0.1;
        if (_useScaleResolution) {
            _filterProgram->setUniformValue("iResolution", _scaleResolution);
        } else {
            _filterProgram->setUniformValue("iResolution", Vector2(_framebuffer->getWidth(), _framebuffer->getHeight()));
        }
        
        
        if (_uniform_mvp != -1) {
            _filterProgram->setUniformValue(_uniform_mvp, _mvp_matrix);
        }
        CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER,  _context->vertexArray));
        CHECK_GL(glClearColor(_backgroundColor.r, _backgroundColor.g, _backgroundColor.b, _backgroundColor.a));
        CHECK_GL(glClear(GL_COLOR_BUFFER_BIT));
        
        int elementIndex = 0;
        for (std::map<int, InputFrameBufferInfo>::const_iterator it = _inputFramebuffers.begin(); it != _inputFramebuffers.end(); ++it) {
            int texIdx = it->first;
            Framebuffer* fb = it->second.frameBuffer;
            if (fb == NULL) {
                return false;
            }
            CHECK_GL(glActiveTexture(GL_TEXTURE0 + texIdx));
            CHECK_GL(glBindTexture(GL_TEXTURE_2D, fb->getTexture()));
            _filterProgram->setUniformValue(
                                            texIdx == 0 ? "colorMap" : str_format("colorMap%d", texIdx),
                                            texIdx);
            // texcoord attribute
            GLuint filterTexCoordAttribute = _filterProgram->getAttribLocation(texIdx == 0 ? "texCoord" : str_format("texCoord%d", texIdx));
            if (filterTexCoordAttribute != (GLuint)-1)
            {
                CHECK_GL(glVertexAttribPointer(filterTexCoordAttribute, 2, GL_FLOAT, 0, 4 * sizeof(GLfloat), (void *)8));
                CHECK_GL(glEnableVertexAttribArray(filterTexCoordAttribute));
            }
            
            elementIndex = it->second.rotationMode;
            
            
        }
        CHECK_GL(glVertexAttribPointer(_filterPositionAttribute, 2, GL_FLOAT, 0, 4 * sizeof(GLfloat), (void *)0));
        CHECK_GL(glEnableVertexAttribArray(_filterPositionAttribute));
        
        //    CHECK_GL(glEnable(GL_BLEND));
        //    CHECK_GL(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
        
        CHECK_GL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,  _context->elementArray[elementIndex]));
        CHECK_GL(glDrawElements(GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_SHORT, 0));
        
        CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, 0));
        CHECK_GL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
        filter_externDraw();
        _framebuffer->inactive();
//        Log("Filter", "%s渲染完毕，准备开始Unlock Framebuffer:%s", typeid(*this).name(),
//            _framebuffer->_hashCode.c_str());
#if DEBUG
		_framebuffer->unlock(typeid(*this).name());
#else
		_framebuffer->unlock();
#endif
        unPrepear();
    }
    return Source::proceed(frameTime, bUpdateTargets);
}

void Filter::filter_externDraw()
{
    
}

const GLfloat* Filter::_getTexureCoordinate(const RotationMode& rotationMode) const {
    static const GLfloat noRotationTextureCoordinates[] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
    };
    
    static const GLfloat rotateLeftTextureCoordinates[] = {
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 0.0f,
        0.0f, 1.0f,
    };
    
    static const GLfloat rotateRightTextureCoordinates[] = {
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 1.0f,
        1.0f, 0.0f,
    };
    
    static const GLfloat verticalFlipTextureCoordinates[] = {
        0.0f, 1.0f,
        1.0f, 1.0f,
        0.0f,  0.0f,
        1.0f,  0.0f,
    };
    
    static const GLfloat horizontalFlipTextureCoordinates[] = {
        1.0f, 0.0f,
        0.0f, 0.0f,
        1.0f,  1.0f,
        0.0f,  1.0f,
    };
    
    static const GLfloat rotateRightVerticalFlipTextureCoordinates[] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
    };
    
    static const GLfloat rotateRightHorizontalFlipTextureCoordinates[] = {
        1.0f, 1.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
    };
    
    static const GLfloat rotate180TextureCoordinates[] = {
        1.0f, 1.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        0.0f, 0.0f,
    };
    
    switch (rotationMode) {
        case NoRotation:
            return noRotationTextureCoordinates;
            break;
        case RotateLeft:
            return rotateLeftTextureCoordinates;
            break;
        case RotateRight:
            return rotateRightTextureCoordinates;
            break;
        case FlipVertical:
            return verticalFlipTextureCoordinates;
            break;
        case FlipHorizontal:
            return horizontalFlipTextureCoordinates;
            break;
        case RotateRightFlipVertical:
            return rotateRightVerticalFlipTextureCoordinates;
            break;
        case RotateRightFlipHorizontal:
            return rotateRightHorizontalFlipTextureCoordinates;
            break;
        case Rotate180:
            return rotate180TextureCoordinates;
            break;
        default:
            break;
    }
}

void Filter::update(float frameTime) {
    if (_inputFramebuffers.empty()) return;
    
    if (!_enable) {
        _framebuffer = _inputFramebuffers.begin()->second.frameBuffer;
        Source::updateTargets(frameTime);
        _framebuffer = 0;
        return;
    }

    if (getContext()->isCapturingFrame && this == getContext()->captureUpToFilter) {
        int captureWidth = getContext()->captureWidth;
        int captureHeight = getContext()->captureHeight;

        _framebuffer = getContext()->getFramebufferCache()->fetchFramebuffer(_context, captureWidth, captureHeight);
#if DEBUG
		_framebuffer->lock(typeid(*this).name());
#else
		_framebuffer->lock();
#endif
        proceed(false);
        
        _framebuffer->active();
        getContext()->capturedFrameData = new unsigned char[captureWidth * captureHeight * 4];
        CHECK_GL(glReadPixels(0, 0, captureWidth, captureHeight, GL_RGBA, GL_UNSIGNED_BYTE, getContext()->capturedFrameData));
        _framebuffer->inactive();
#if DEBUG
		_framebuffer->unlock(typeid(*this).name());
#else
		_framebuffer->unlock();
#endif
    } else {
        // todo
        Framebuffer* firstInputFramebuffer = _inputFramebuffers.begin()->second.frameBuffer;
        RotationMode firstInputRotation = _inputFramebuffers.begin()->second.rotationMode;
        if (!firstInputFramebuffer) return;

        int rotatedFramebufferWidth = firstInputFramebuffer->getWidth();
        int rotatedFramebufferHeight = firstInputFramebuffer->getHeight();
        if (rotationSwapsSize(firstInputRotation))
        {
            rotatedFramebufferWidth = firstInputFramebuffer->getHeight();
            rotatedFramebufferHeight = firstInputFramebuffer->getWidth();
        }

        if (_framebufferScale !=  1.0) {
            rotatedFramebufferWidth = int(rotatedFramebufferWidth * _framebufferScale);
            rotatedFramebufferHeight = int(rotatedFramebufferHeight * _framebufferScale);
        }

        _framebuffer = getContext()->getFramebufferCache()->fetchFramebuffer(_context, rotatedFramebufferWidth, rotatedFramebufferHeight);
        proceed(frameTime);
    }
//    _context->getFramebufferCache()->returnFramebuffer(_framebuffer);
    _framebuffer = 0;
}

bool Filter::getProperty(const std::string& name, std::vector<Vec2>& retValue) {
    Property* property = _getProperty(name);
    if (!property) return false;
    retValue = ((Vec2ArrayProperty*)property)->value;
    return true;
}

bool Filter::registerProperty(const std::string& name,
                              std::vector<Vec2> defaultValue,
                              const std::string& comment/* = ""*/,
                              std::function<void(std::vector<Vec2>&)> setCallback/* = 0*/) {
    if (hasProperty(name)) return false;
    Vec2ArrayProperty property;
    property.type = "vec2Array";
    property.value = defaultValue;
    property.comment = comment;
    property.setCallback = setCallback;
    _vec2ArrayProperties[name] = property;
    return true;
}

bool Filter::setProperty(const std::string& name, std::vector<Vec2> value) {
    Property* rawProperty = _getProperty(name);
    if (!rawProperty) {
        Log("WARNING", "Filter::setProperty invalid property %s", name.c_str());
        return false;
    } else if (rawProperty->type != "vec2Array") {
        Log("WARNING", "Filter::setProperty The property type is expected to be %s", rawProperty->type.c_str());
        return false;
    }
    Vec2ArrayProperty* property = ((Vec2ArrayProperty *)rawProperty);
    property->value = value;
    if (property->setCallback)
        property->setCallback(value);
    return true;
}

bool Filter::registerProperty(const std::string& name, int defaultValue, const std::string& comment/* = ""*/, std::function<void(int&)> setCallback/* = 0*/) {
    if (hasProperty(name)) return false;
    IntProperty property;
    property.type = "int";
    property.value = defaultValue;
    property.comment = comment;
    property.setCallback = setCallback;
    _intProperties[name] = property;
    return true;
}

bool Filter::registerProperty(const std::string& name, float defaultValue, const std::string& comment/* = ""*/, std::function<void(float&)> setCallback/* = 0*/) {
    if (hasProperty(name)) return false;
    FloatProperty property;
    property.type = "float";
    property.value = defaultValue;
    property.comment = comment;
    property.setCallback = setCallback;
    _floatProperties[name] = property;
    return true;
}

bool Filter::registerProperty(const std::string& name, const std::string& defaultValue, const std::string& comment/* = ""*/, std::function<void(std::string&)> setCallback/* = 0*/) {
    if (hasProperty(name)) return false;
    StringProperty property;
    property.type = "string";
    property.value = defaultValue;
    property.comment = comment;
    property.setCallback = setCallback;
    _stringProperties[name] = property;
    return true;
}

bool Filter::setProperty(const std::string& name, int value) {
    Property* rawProperty = _getProperty(name);
    if (!rawProperty) {
        Log("WARNING", "Filter::setProperty invalid property %s", name.c_str());
        return false;
    } else if (rawProperty->type != "int") {
        Log("WARNING", "Filter::setProperty The property type is expected to be %s", rawProperty->type.c_str());
        return false;
    }
    IntProperty* property = ((IntProperty*)rawProperty);
    property->value = value;
    if (property->setCallback)
        property->setCallback(value);
    return true;
}

bool Filter::setProperty(const std::string& name, float value) {
    Property* rawProperty = _getProperty(name);
    if (!rawProperty) {
        Log("WARNING", "Filter::setProperty invalid property %s", name.c_str());
        return false;
    } else if (rawProperty->type != "float") {
        Log("WARNING", "Filter::setProperty The property type is expected to be %s", rawProperty->type.c_str());
        return false;
    }
    FloatProperty* property = ((FloatProperty*)rawProperty);
    if (property->setCallback)
        property->setCallback(value);
    property->value = value;

    return true;
}

bool Filter::setProperty(const std::string& name, std::string value) {
    Property* rawProperty = _getProperty(name);
    if (!rawProperty) {
        Log("WARNING", "Filter::setProperty invalid property %s", name.c_str());
        return false;
    } else if (rawProperty->type != "string") {
        Log("WARNING", "Filter::setProperty The property type is expected to be %s", rawProperty->type.c_str());
        return false;
    }
    StringProperty* property = ((StringProperty*)rawProperty);
    property->value = value;
    if (property->setCallback)
        property->setCallback(value);
    return true;
}

bool Filter::getProperty(const std::string& name, int& retValue) {
    Property* property = _getProperty(name);
    if (!property) return false;
    retValue = ((IntProperty*)property)->value;
    return true;
}

bool Filter::getProperty(const std::string& name, float& retValue) {
    Property* property = _getProperty(name);
    if (!property) return false;
    retValue = ((FloatProperty*)property)->value;
    return true;
}

bool Filter::getProperty(const std::string& name, std::string& retValue) {
    Property* property = _getProperty(name);
    if (!property) return false;
    retValue = ((StringProperty*)property)->value;
    return true;
}

Filter::Property* Filter::_getProperty(const std::string& name) {
    if (_intProperties.find(name) != _intProperties.end()) {
        return &_intProperties[name];
    }
    if (_floatProperties.find(name) != _floatProperties.end()) {
        return &_floatProperties[name];
    }
    if (_stringProperties.find(name) != _stringProperties.end()) {
        return &_stringProperties[name];
    }
    
    if (_vec2ArrayProperties.find(name) != _vec2ArrayProperties.end()) {
        return &_vec2ArrayProperties[name];
    }
    
    if (_vec2Properties.find(name) != _vec2Properties.end()) {
        return &_vec2Properties[name];
    }
    
    return 0;
}

bool Filter::hasProperty(const std::string& name, const std::string type) {
    Property* property = _getProperty(name);
    return property && property->type == type ? true : false;
}

bool Filter::hasProperty(const std::string& name) {
     return _getProperty(name) ? true : false;
}

bool Filter::getPropertyComment(const std::string& name, std::string& retComment) {
    Property* property = _getProperty(name);
    if (!property) return false;
    retComment = std::string("[") + property->type + "] " + property->comment;
    return true;
}

bool Filter::getPropertyType(const std::string& name, std::string& retType) {
    Property* property = _getProperty(name);
    if (!property) return false;
    retType = property->type;
    return true;
}

Context *Filter::getContext() {
    if (_context) {
        return _context;
    }
    
    return NULL;
}
NS_GI_END

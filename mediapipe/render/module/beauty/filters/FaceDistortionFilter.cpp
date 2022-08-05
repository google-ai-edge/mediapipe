#include "FaceDistortionFilter.hpp"

namespace Opipe
{
    const std::string kFaceDistortionVertexShaderString = SHADER_STRING(
        precision highp float;
        attribute vec4 texCoord;
        varying vec2 vTexCoord;
        uniform float aspectRatio;
        uniform vec2 center[20];
        uniform vec2 radius[20];

        uniform float scale[20];
        uniform float angle[20];
        uniform float u_min[20];
        uniform float u_max[20];
        uniform int types[20];
        uniform int count;
        uniform float eye;
        uniform float slim;
        uniform float nose;
        uniform int debug;
        void main() {
            vec2 uv = texCoord.xy;
            gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
            for (int i = 0; i < count; i++)
            {
                if (scale[i] == 0.0 || types[i] == 0)
                {
                    continue;
                }
                vec2 textureCoordinateToUse = uv;
                float e1 = (textureCoordinateToUse.x - center[i].x) / (radius[i].x);
                float e2 = (textureCoordinateToUse.y - center[i].y) / (radius[i].y / aspectRatio);
                float d = (e1 * e1) + (e2 * e2);
                if (d < 1.0)
                {
                    if (types[i] == 1)
                    {
                        vec2 dist = vec2(d * radius[i].x, d * radius[i].y);
                        textureCoordinateToUse -= center[i];
                        vec2 delta = ((radius[i] - dist) / radius[i]);
                        float deltaScale = scale[i];
                        if (deltaScale > 0.0)
                        {
                            deltaScale = smoothstep(u_min[i], u_max[i], deltaScale);
                        }
                        vec2 percent = 1.0 - ((delta * deltaScale) * eye);
                        textureCoordinateToUse = textureCoordinateToUse * percent;
                        uv = textureCoordinateToUse + center[i];
                    }
                    else if (types[i] == 2)
                    {
                        float dist = 1.0 - d;
                        float delta = scale[i] * dist * slim;
                        float deltaScale = smoothstep(u_min[i], u_max[i], dist);
                        float directionX = cos(angle[i]) * deltaScale;
                        float directionY = sin(angle[i]) * deltaScale / (3.0 / 4.0 * aspectRatio);
                        uv = vec2(textureCoordinateToUse.x - (delta * directionX),
                                  textureCoordinateToUse.y - (delta * directionY));
                    }
                    else if (types[i] == 3)
                    {
                        float dist = 1.0 - d;
                        float delta = scale[i] * dist * nose;
                        float deltaScale = smoothstep(u_min[i], u_max[i], dist);
                        float directionX = cos(angle[i]) * deltaScale;
                        float directionY = sin(angle[i]) * deltaScale / (3.0 / 4.0 * aspectRatio);
                        uv = vec2(textureCoordinateToUse.x - (delta * directionX),
                                  textureCoordinateToUse.y - (delta * directionY));
                    }
                }
            }
            vTexCoord = uv;
        });

    const std::string kFaceDistortionFragmentShaderString = SHADER_STRING(
        precision highp float;
        uniform sampler2D colorMap;
        varying vec2 vTexCoord;
        uniform vec2 facePoints[106];
        void main() {
            highp vec4 textureColor = texture2D(colorMap, vTexCoord);
            gl_FragColor = textureColor;
        });

    FaceDistortionFilter::FaceDistortionFilter(Context *context) : Filter(context)
    {
    }

    FaceDistortionFilter::~FaceDistortionFilter()
    {
        releaseDistoritionVBO();
    }

    void FaceDistortionFilter::generateDistoritionVBO(int numX,
                                                      int numY,
                                                      const GLfloat *imageTexUV)
    {
        if (vao == -1)
        {
            CHECK_GL(glGenBuffers(1, &vao));
            CHECK_GL(glGenBuffers(1, &eao));

            int vCount = numX * (numY + 1) * 2;
            GLfloat *divideImageTexUV = (GLfloat *)malloc(vCount * 2 * sizeof(GLfloat));
            GLushort *element = (GLushort *)malloc((vCount + numX - 1) * sizeof(GLushort));

            float offsetX = ((imageTexUV[2] - imageTexUV[0]) / numX);
            float offsetY = ((imageTexUV[5] - imageTexUV[1]) / numY);
            int elementIndex = 0;
            for (int i = 0; i < numX; i++)
            {
                for (int j = 0; j <= numY; j++)
                {
                    int offset = (i * (numY + 1) + j) * 4;

                    divideImageTexUV[offset] = imageTexUV[0] + i * offsetX;
                    divideImageTexUV[offset + 1] = imageTexUV[1] + j * offsetY;
                    divideImageTexUV[offset + 2] = divideImageTexUV[offset] + offsetX;
                    divideImageTexUV[offset + 3] = divideImageTexUV[offset + 1];

                    element[elementIndex++] = offset / 2;
                    element[elementIndex++] = offset / 2 + 1;
                }
                if (elementIndex < (vCount + numX - 1))
                {
                    element[elementIndex++] = 0xFFFF;
                }
            }
            CHECK_GL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eao));
            CHECK_GL(glBufferData(GL_ELEMENT_ARRAY_BUFFER, (vCount + numX - 1) * sizeof(GLushort),
                                  element, GL_STATIC_DRAW));

            CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, vao));
            CHECK_GL(glBufferData(GL_ARRAY_BUFFER, vCount * 2 * sizeof(GLfloat),
                                  divideImageTexUV,
                                  GL_STATIC_DRAW));
            free(element);
            free(divideImageTexUV);

            CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, 0));
            CHECK_GL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
        }
    }

    void FaceDistortionFilter::releaseDistoritionVBO()
    {
        if (vao != -1)
        {
            CHECK_GL(glDeleteBuffers(0, &vao));
            vao = -1;
            CHECK_GL(glDeleteBuffers(0, &eao));
            eao = -1;
        }
    }

    FaceDistortionFilter *FaceDistortionFilter::create(Context *context)
    {
        FaceDistortionFilter *ret =
            new (std::nothrow) FaceDistortionFilter(context);
        if (!ret || !ret->init(context))
        {
            delete ret;
            ret = 0;
        }
        return ret;
    }

    bool FaceDistortionFilter::init(Context *context)
    {
        if (!initWithShaderString(context,
                                  kFaceDistortionVertexShaderString,
                                  kFaceDistortionFragmentShaderString))
        {
            return false;
        }
        _eye = 0.0;
        _slim = 0.0;
        return true;
    }

    void FaceDistortionFilter::addPoint(Vector2 center,
                                        float radiusX,
                                        float radiusY,
                                        float scale,
                                        int type,
                                        float angle,
                                        float min,
                                        float max)
    {
        _center[_count * 2] = center.x / 2 + 0.5;
        _center[_count * 2 + 1] = center.y / 2 + 0.5;
        _radius[_count * 2] = radiusX;
        _radius[_count * 2 + 1] = radiusY;
        _scale[_count] = scale;
        _angle[_count] = angle;
        _types[_count] = type;
        _u_min[_count] = min;
        _u_max[_count] = max;
        _count++;
    }

    float getRadius(Vector2 anglePoint, Vector2 center)
    {
        float angle = 0;
        if (fabs(anglePoint.x - center.x) <= 0.00001)
        {
            angle = anglePoint.y > center.y ? M_PI_2 : -M_PI_2;
        }
        else if (fabs(anglePoint.y - center.y) <= 0.00001)
        {
            angle = anglePoint.x > center.x ? 0 : M_PI;
        }
        else
        {
            float radius = (anglePoint.y - center.y) / (anglePoint.x - center.x);
            angle = atanf(radius);
            if ((angle > 0 && anglePoint.y - center.y < 0) ||
                (angle < 0 && anglePoint.y - center.y > 0))
            {
                angle += M_PI;
            }
        }
        return angle;
    }

    Vector2 FaceDistortionFilter::_positionAt(int index) {
        float x = (_facePoints[index].x - 0.5) * 2.0;
        float y = (_facePoints[index].y - 0.5) * 2.0;
        return Vector2(x, y);
    }

    void FaceDistortionFilter::setUniform()
    {
        if (_facePoints.size() > 60)
        {

            _count = 0;
            float width = (float)getFramebuffer()->getWidth();
            float height = (float)getFramebuffer()->getHeight();
            _filterProgram->setUniformValue("aspectRatio",
                                            height /
                                                width);
            _filterProgram->setUniformValue("eye", _eye);
            _filterProgram->setUniformValue("slim", _slim);
            _filterProgram->setUniformValue("nose", _nose);
            //左眼放大
            {
                Vector2 point1 = _positionAt(362);
                Vector2 point2 = _positionAt(263);
                Vector2 point3 = _positionAt(168);
                Vector2 center = point1.getCenter(point2);
                float distance = center.distance(point3);
                addPoint(center, distance / 2, distance / 2, 0.3, 1, 0.0f, 0.0f, 1);
            }

            //右眼放大
            {
                Vector2 point1 = _positionAt(33);
                Vector2 point2 = _positionAt(133);
                Vector2 point3 = _positionAt(168);
                Vector2 center = point1.getCenter(point2);
                float distance = center.distance(point3);
                addPoint(center, distance / 2, distance / 2, 0.3, 1, 0.0f, 0.0f, 1);
            }
            //瘦左脸
            {

                Vector2 point1 = _positionAt(136);
                Vector2 point2 = _positionAt(19);
                Vector2 point3 = _positionAt(234);
                Vector2 point4 = _positionAt(152);

                float angle = getRadius(point2, point1);
                addPoint(point1, point1.distance(point3), point1.distance(point4), 0.02, 2, angle,
                         0.0f,
                         0.02);
            }
            //瘦右脸
            {
                Vector2 point1 = _positionAt(379);
                Vector2 point2 = _positionAt(19);
                Vector2 point3 = _positionAt(454);
                Vector2 point4 = _positionAt(152);

                float angle = getRadius(point2, point1);
                addPoint(point1, point1.distance(point3), point1.distance(point4), 0.02, 2, angle,
                         0.0f,
                         0.02);
            }
            
            //瘦左鼻子
            {
                Vector2 point1 = _positionAt(219);
                Vector2 point2 = _positionAt(4);
                Vector2 point3 = _positionAt(131);
                Vector2 point4 = _positionAt(60);

                float angle = getRadius(point2, point1);
                addPoint(point1, point1.distance(point3), point1.distance(point4), 0.02, 3, angle,
                         0.0f,
                         0.02);
            }
            
            //瘦右鼻子鼻子
            {
                Vector2 point1 = _positionAt(294);
                Vector2 point2 = _positionAt(4);
                Vector2 point3 = _positionAt(429);
                Vector2 point4 = _positionAt(290);

                float angle = getRadius(point2, point1);
                addPoint(point1, point1.distance(point3), point1.distance(point4), 0.02, 3, angle,
                         0.0f,
                         0.02);
            }

            _filterProgram->setUniformValue("count", _count);
            _filterProgram->setUniformValue("center", _count, _center, 2);
            _filterProgram->setUniformValue("radius", _count, _radius, 2);
            _filterProgram->setUniformValue("facePoints", (int)_facePoints.size(),
                                            _u_facePoints, 2);

            _filterProgram->setUniformValue("angle", _count, _angle);
            _filterProgram->setUniformValue("scale", _count, _scale);
            _filterProgram->setUniformValue("u_min", _count, _u_min);
            _filterProgram->setUniformValue("u_max", _count, _u_max);
            _filterProgram->setUniformValue("types", _count, _types);
        }
        else
        {
            _filterProgram->setUniformValue("count", 0);
        }
    }

    bool FaceDistortionFilter::proceed(float frameTime, bool bUpdateTargets)
    {
#if DEBUG
        _framebuffer->lock(typeid(*this).name());
#else
        _framebuffer->lock();
#endif
        setUniform();
        _context->setActiveShaderProgram(_filterProgram);

        _framebuffer->active();
        CHECK_GL(glClearColor(_backgroundColor.r, _backgroundColor.g,
                              _backgroundColor.b,
                              _backgroundColor.a));
        CHECK_GL(glClear(GL_COLOR_BUFFER_BIT));

        const int numX = 20;
        const int numY = 20;
        for (std::map<int, InputFrameBufferInfo>::const_iterator it = _inputFramebuffers.begin();
             it != _inputFramebuffers.end(); ++it)
        {
            int texIdx = it->first;
            Framebuffer *fb = it->second.frameBuffer;
            CHECK_GL(glActiveTexture(GL_TEXTURE0 + texIdx));
            CHECK_GL(glBindTexture(GL_TEXTURE_2D, fb->getTexture()));
            _filterProgram->setUniformValue(
                texIdx == 0 ? "colorMap" : str_format("colorMap%d", texIdx),
                texIdx);
            // texcoord attribute
            GLuint filterTexCoordAttribute =
                _filterProgram->getAttribLocation(texIdx == 0 ? "texCoord" : str_format("texCoord%d", texIdx));
            CHECK_GL(glEnableVertexAttribArray(filterTexCoordAttribute));
            if (texIdx == 0)
            {
                const GLfloat *imageTexUV = _getTexureCoordinate(it->second.rotationMode);
                generateDistoritionVBO(numX, numY, imageTexUV);
                CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, vao));
            }
            CHECK_GL(glVertexAttribPointer(filterTexCoordAttribute, 2, GL_FLOAT, 0,
                                           2 * sizeof(GLfloat), (void *)0));
        }

        CHECK_GL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eao));
        CHECK_GL(glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX));
        CHECK_GL(glDrawElements(GL_TRIANGLE_STRIP, numX * (numY + 1) * 2 + numX - 1,
                                GL_UNSIGNED_SHORT, (void *)0));

        CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, 0));
        CHECK_GL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
        filter_externDraw();
        _framebuffer->inactive();
#if DEBUG
        _framebuffer->unlock(typeid(*this).name());
#else
        _framebuffer->unlock();
#endif
        unPrepear();
        return Source::proceed(frameTime, bUpdateTargets);
    }

}

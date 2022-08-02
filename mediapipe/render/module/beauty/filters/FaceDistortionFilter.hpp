#ifndef FaceDistortionFilter_hpp
#define FaceDistortionFilter_hpp

#include <stdio.h>
#include "mediapipe/render/core/math/vec2.hpp"
#include "mediapipe/render/core/Filter.hpp"
#include "mediapipe/render/core/Context.hpp"

namespace Opipe
{
    class FaceDistortionFilter : public virtual Filter
    {
    public:
        virtual ~FaceDistortionFilter();
        FaceDistortionFilter(Context *context);

        static FaceDistortionFilter *create(Context *context);
        bool init(Context *context);
        virtual bool proceed(float frameTime = 0.0, bool bUpdateTargets = true) override;

    public:
        float eye() {
            return _eye;
        }
        
        float slim() {
            return _slim;
        }
        
        float nose() {
            return _nose;
        }
        
        void setEye(float eye)
        {
            _eye = eye;
        }

        void setSlim(float slim)
        {
            _slim = slim;
        }
        
        void setNose(float nose)
        {
            _nose = nose;
        }

        void setFacePoints(std::vector<Vec2> facePoints)
        {
            _facePoints = facePoints;
            for (int i = 0; i < _facePoints.size(); i++)
            {
                auto facePoint = _facePoints[i];
                _u_facePoints[i * 2] = facePoint.x;
                _u_facePoints[i * 2 + 1] = facePoint.y;
            }
        }

    private:
        void setUniform();
        void addPoint(Vector2 center, float radiusX, float radiusY,
                      float scale, int type,
                      float angle, float min = 0.0f, float max = 1.0f);
        int _count;
        float _center[40];
        float _radius[40];
        float _scale[20];
        float _angle[20];
        float _u_min[20];
        float _u_max[20];
        int _types[20];
        float _u_facePoints[980];
        
        Vector2 _positionAt(int index);

    private:
        void generateDistoritionVBO(int numX, int numY, const GLfloat *imageTexUV);
        void releaseDistoritionVBO();

    private:
        float _eye = 0.0;
        float _slim = 0.0;
        float _nose = 0.0;
        std::vector<Vec2> _facePoints; //暂时支持单个人脸
        GLuint vao = -1;
        GLuint eao = -1;
    };
}

#endif

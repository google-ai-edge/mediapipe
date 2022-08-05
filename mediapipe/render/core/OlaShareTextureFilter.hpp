//
//  OlaShareTextureFilter.hpp
//  AREmotion
//
//  Created by Renzhu Wang on 07/12/2017.
//  Copyright © 2022 olachat. All rights reserved.
//

#ifndef OlaShareTextureFilter_hpp
#define OlaShareTextureFilter_hpp

#include "Filter.hpp"
#include "Context.hpp"


namespace Opipe {
    class OlaShareTextureFilter : public Opipe::Filter {
    public:
        static OlaShareTextureFilter* create(Opipe::Context *context);
        static OlaShareTextureFilter* create(Opipe::Context *context, GLuint targetTextureId, TextureAttributes attributes);
        bool init(Opipe::Context *context);
        virtual bool proceed(float frameTime = 0, bool bUpdateTargets = true) override;
        virtual void update(float frameTime) override;
        void updateTargetId(GLuint targetId);
        
        GLuint targetTextureId;
        
        
    public:
        OlaShareTextureFilter(Opipe::Context *context);
        virtual ~OlaShareTextureFilter() noexcept;
        
        TextureAttributes targetTextureAttr = Framebuffer::defaultTextureAttribures;
    private:
        bool _targetFramebuffer = false;
    };
}

#endif /* OlaShareTextureFilter_hpp */

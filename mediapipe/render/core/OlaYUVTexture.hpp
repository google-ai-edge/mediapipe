#ifndef Ola_YUVTexture_hpp
#define Ola_YUVTexture_hpp

#include "Filter.hpp"

NS_GI_BEGIN
class OlaYUVTexture : public Opipe::Filter {
public:
    static OlaYUVTexture* create(Opipe::Context *context);
    bool init(Opipe::Context *context);
public:
    ~OlaYUVTexture();
    OlaYUVTexture(Opipe::Context *context);
    Opipe::Context *_context;
};
NS_GI_END
#endif /* Ola_YUVTexture_hpp */

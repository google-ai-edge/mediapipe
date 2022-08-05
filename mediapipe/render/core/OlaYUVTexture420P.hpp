#ifndef OlaYUVTexture420P_hpp
#define OlaYUVTexture420P_hpp

#include "Filter.hpp"

NS_GI_BEGIN
class OlaYUVTexture420P : public Opipe::Filter {
public:
    static OlaYUVTexture420P* create(Opipe::Context *context);
    bool init(Opipe::Context *context);
public:
    ~OlaYUVTexture420P();
    OlaYUVTexture420P(Opipe::Context *context);
    Opipe::Context *_context;
};
NS_GI_END
#endif /* OlaYUVTexture420P_hpp */

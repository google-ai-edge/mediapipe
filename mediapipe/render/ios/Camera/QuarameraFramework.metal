//
//  WebARShader.metal
//  Pods-WebAR-iOS
//
//  Created by Wang Huai on 2019/9/16.
//

#include <metal_stdlib>
using namespace metal;

struct SingleInputVertexIO
{
    float4 position [[position]];
    float2 textureCoordinate [[user(texturecoord)]];
};

struct TextureScale
{
    float3 scale;
};

struct TwoInputVertexIO
{
    float4 position [[position]];
    float2 textureCoordinate [[user(texturecoord)]];
    float2 textureCoordinate2 [[user(texturecoord2)]];
};

fragment half4 yuvConversionFullRangeFragment(TwoInputVertexIO fragmentInput [[stage_in]],
                                              texture2d<half> inputTexture [[texture(0)]],
                                              texture2d<half> inputTexture2 [[texture(1)]],
                                              constant TextureScale &scaleUniform [[buffer(0)]])
{
    constexpr sampler quadSampler;
    float2 scaleCoordinate = (fragmentInput.textureCoordinate - 0.5) * scaleUniform.scale.xy + 0.5;
    if(scaleUniform.scale.z == 1.0) {
        scaleCoordinate.y = 1.0 - scaleCoordinate.y;
    }
//    scaleCoordinate.y = -(scaleUniform.scale.z + scaleCoordinate.y);
    half3 yuv;
    yuv.x = inputTexture.sample(quadSampler, scaleCoordinate).r;
    yuv.yz = inputTexture2.sample(quadSampler, scaleCoordinate).rg - half2(0.5, 0.5);
    const half3x3 kColorConversion601Default = {
        {1.164,  1.164,  1.164},
        {0.0,    -0.392, 2.017},
        {1.596,  -0.813, 0.0},
    };
    
    half3 rgb = kColorConversion601Default * yuv;
    
    return half4(rgb, 1.0);
}



vertex TwoInputVertexIO twoInputVertex(const device packed_float2 *position [[buffer(0)]],
                                       const device packed_float2 *texturecoord [[buffer(1)]],
                                       const device packed_float2 *texturecoord2 [[buffer(2)]],
                                       uint vid [[vertex_id]])
{
    TwoInputVertexIO outputVertices;
    
    outputVertices.position = float4(position[vid], 0, 1.0);
    outputVertices.textureCoordinate = texturecoord[vid];
    outputVertices.textureCoordinate2 = texturecoord2[vid];
    
    return outputVertices;
}

constant half3 ColorOffsetFullRange = half3(0.0, -0.5, -0.5);

kernel void yuv2rgb(texture2d<half, access::read> y_tex    [[ texture(0) ]],
                    texture2d<half, access::read> uv_tex   [[ texture(1) ]],
                    texture2d<half, access::write> bgr_tex   [[ texture(2) ]],
                    uint2 gid [[thread_position_in_grid]])
{
    
    half3 yuv = half3(y_tex.read(gid).r, half2(uv_tex.read(gid/2).rg)) + ColorOffsetFullRange;
    
    const half3x3 kColorConversion601Default = {
        {1.164,  1.164,  1.164},
        {0.0,  -0.392, 2.017},
        {1.596,  -0.813, 0.0},
    };
    
    half4 conversion = half4(kColorConversion601Default * yuv, 1.0);
    bgr_tex.write(conversion, gid);
}



vertex SingleInputVertexIO oneInputVertex(const device packed_float2 *position [[buffer(0)]],
                                          const device packed_float2 *texturecoord [[buffer(1)]],
                                          uint vid [[vertex_id]])
{
    SingleInputVertexIO outputVertices;
    
    outputVertices.position = float4(position[vid], 0, 1.0);
    outputVertices.textureCoordinate = texturecoord[vid];
    
    return outputVertices;
}

float udRoundBox(half2 p, half2 b, float r)
{
    
    return length(max(abs(p) - b + r, 0.0)) - r;
}

typedef struct
{
    int useScan;
    float iRadius;
    float squareWidth;
    float width;
    float height;
    float iTime;
} ScanUniform;

fragment half4 scanFragment(SingleInputVertexIO fragmentInput [[stage_in]],
                                   texture2d<half> inputTexture [[texture(0)]],
                                   constant ScanUniform& uniform [[buffer(1)]])
{
    constexpr sampler quadSampler(mag_filter::linear,
                                  min_filter::linear);
    half4 color = inputTexture.sample(quadSampler, fragmentInput.textureCoordinate);
    if (uniform.useScan == 0) {
        return color;
    }
    
    half2 iResolution = half2(uniform.width, uniform.height);
    half2 uv = half2(fragmentInput.textureCoordinate);
    half2 fragCoord = iResolution * uv;
    half2 center = iResolution / 2.0;
    half2 halfRes = 0.5 * half2(uniform.squareWidth);
    float b = udRoundBox(fragCoord.xy - center, halfRes, uniform.iRadius);
    color = half4(mix(color.rgb, half3(0.0), smoothstep(0.0, 1.0, b) * 0.5 ).rgb, 1.0);
    
    return color;
}

fragment half4 passthroughFragment(SingleInputVertexIO fragmentInput [[stage_in]],
                                   texture2d<half> inputTexture [[texture(0)]])
{
    constexpr sampler quadSampler(mag_filter::linear,
                                  min_filter::linear);
    half4 color = inputTexture.sample(quadSampler, fragmentInput.textureCoordinate);
    return color;
}


typedef struct {
    float4 position [[position]];
    float2 texCoord;
} WAFrameVertexOut;

constant half3 c0 (1.,  1., 1.);
constant half3 c1 (0., -.18732, 1.8556);
constant half3 c2 (1.57481, -.46813,0.);

vertex WAFrameVertexOut v_display(uint vertexID [[ vertex_id ]],
                                  constant packed_float4* vertexArray [[ buffer(0) ]]) {
    float4 vertexData = vertexArray[vertexID];
    WAFrameVertexOut out;
    out.position = float4(vertexData.xy,0.0,1.0);
    out.texCoord = vertexData.zw;
    
    return out;
}

fragment float4 f_display(WAFrameVertexOut in [[stage_in]],
                          texture2d<half, access::sample> colorTexture [[texture(0)]]) {
    
    constexpr sampler textureSampler (mag_filter::linear,
                                      min_filter::linear);
    
    // Sample the texture to obtain a color
    const half4 colorSample = colorTexture.sample(textureSampler, in.texCoord);
    
    // return the color of the texture
    return float4(colorSample);
}

kernel void k_ycr2rgb(texture2d<half, access::read> lumTexture [[texture(0)]],
                      texture2d<half, access::read> chromeTexture [[texture(1)]],
                      texture2d<half, access::write> outTexture [[texture(2)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if ((gid.x >= lumTexture.get_width()) || (gid.y >= lumTexture.get_height())) {
        return;
    }
    
    half3 ycbcr = half3(lumTexture.read(gid).r , chromeTexture.read(gid/2).rg - 0.5);
    half3 color = half3x3(c0,c1,c2) * ycbcr;
    
    outTexture.write(half4(color, 1.0), gid);
}

precision highp float;
precision highp int;
varying vec2 textureCoordinate;
uniform sampler2D inputImageTexture;
uniform float inputImageTextureWidth;
uniform float inputImageTextureHeight;
float cY(float x,float y){
    vec4 c=texture2D(inputImageTexture,vec2(x,y));
    return 0.183 * c.r + 0.614 * c.g + 0.062 * c.b + 0.0625;
}

vec4 cC(float x,float y,float dx,float dy){
    vec4 c0=texture2D(inputImageTexture,vec2(x,y));
    vec4 c1=texture2D(inputImageTexture,vec2(x+dx,y));
    vec4 c2=texture2D(inputImageTexture,vec2(x,y+dy));
    vec4 c3=texture2D(inputImageTexture,vec2(x+dx,y+dy));
    return (c0+c1+c2+c3)/4.;
}

float cU(float x,float y,float dx,float dy){
    vec4 c=cC(x,y,dx,dy);
    return -0.101 * c.r - 0.339 * c.g + 0.439 * c.b + 0.5000;
}

float cV(float x,float y,float dx,float dy){
    vec4 c=cC(x,y,dx,dy);
    return 0.439 * c.r - 0.399 * c.g - 0.040 * c.b + 0.5000;
}

vec2 cPos(float t,float shiftx,float gy){
    vec2 pos=vec2(floor(inputImageTextureWidth*textureCoordinate.x),floor(inputImageTextureHeight*gy));
    return vec2(mod(pos.x*shiftx,inputImageTextureWidth),(pos.y*shiftx+floor(pos.x*shiftx/inputImageTextureWidth))*t);
}

vec4 calculateY(){
    vec2 pos=cPos(1.,4.,textureCoordinate.y);
    vec4 oColor=vec4(0);
    float textureYPos=pos.y/inputImageTextureHeight;
    oColor[0]=cY(pos.x/inputImageTextureWidth,textureYPos);
    oColor[1]=cY((pos.x+1.)/inputImageTextureWidth,textureYPos);
    oColor[2]=cY((pos.x+2.)/inputImageTextureWidth,textureYPos);
    oColor[3]=cY((pos.x+3.)/inputImageTextureWidth,textureYPos);
    return oColor;
}
vec4 calculateUV(float dx,float dy){
    vec2 pos=cPos(2.,4.,textureCoordinate.y-0.2500);
    vec4 oColor=vec4(0);
    float textureYPos=pos.y/inputImageTextureHeight;
    oColor[0]= cV(pos.x/inputImageTextureWidth,textureYPos,dx,dy);
    oColor[1]= cU(pos.x/inputImageTextureWidth,textureYPos,dx,dy);
    oColor[2]= cV((pos.x+2.)/inputImageTextureWidth,textureYPos,dx,dy);
    oColor[3]= cU((pos.x+2.)/inputImageTextureWidth,textureYPos,dx,dy);
    return oColor;
}
void main() {
    if(textureCoordinate.y<0.2500){
        gl_FragColor=calculateY();
    }else if(textureCoordinate.y<0.3750){
        gl_FragColor=calculateUV(1./inputImageTextureWidth,1./inputImageTextureHeight);
    }else{
        gl_FragColor=vec4(0,0,0,0);
    }
}
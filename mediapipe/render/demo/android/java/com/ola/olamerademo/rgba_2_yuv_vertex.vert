    attribute vec4 position;
    attribute vec4 inputTextureCoordinate;
    uniform mat4 mvp;

    varying vec2 textureCoordinate;

    void main()
    {
        gl_Position = mvp * position;
        textureCoordinate = inputTextureCoordinate.xy;
    }
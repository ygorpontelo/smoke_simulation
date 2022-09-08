attribute vec2 position;
attribute float densitypoint;

varying density;

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
}

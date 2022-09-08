uniform vec3 FillColor;

input float density;

void main()
{
    gl_FragColor = vec4(FillColor, density);
}

#include "misc/spatial-filters.frag"

uniform vec3 FillColor;

uniform sampler2D density;
uniform vec2 scale;


void main()
{
    float v = texture2D(density, gl_FragCoord.xy*scale).r;
    gl_FragColor = vec4(FillColor, v);
}

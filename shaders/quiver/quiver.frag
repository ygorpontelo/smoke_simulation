#include "math/constants.glsl"
#include "arrows/arrows.glsl"
#include "antialias/antialias.glsl"

uniform vec2 iResolution;
uniform float linewidth;

uniform int dim_x;
uniform int dim_y;

uniform sampler2D velocities;

out vec4 fragcolor;


void main()
{
    const float M_PI = 3.14159265358979323846;
    const float SQRT_2 = 1.4142135623730951;
    const float antialias = 1.0;

    float body = min(iResolution.x/dim_x, iResolution.y/dim_y) / SQRT_2;
    vec2 texcoord = gl_FragCoord.xy;
    vec2 size   = iResolution / vec2(dim_y, dim_x);
    vec2 center = (floor(texcoord/size) + vec2(0.5, 0.5)) * size;

    texcoord -= center;

    float idrow = gl_FragCoord.x / size.x;
    float idcol = (iResolution.y - gl_FragCoord.y) / size.y; 

    vec2 pos = vec2(idrow, idcol) * vec2(1.0/dim_x, 1.0/dim_y);
    float u = texture2D(velocities, pos).r;
    float v = texture2D(velocities, pos).g;

    float theta = M_PI-atan(v, -u);
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    texcoord = vec2(cos_theta*texcoord.x - sin_theta*texcoord.y,
                    sin_theta*texcoord.x + cos_theta*texcoord.y);

    float d = arrow_stealth(texcoord, body, 0.25*body, linewidth, antialias);
    fragcolor = filled(d, linewidth, antialias, vec4(1,1,1,1));
}

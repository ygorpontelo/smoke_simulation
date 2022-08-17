uniform vec2 iResolution;
uniform vec2 dimensions;
uniform float thickness;

void main() {
    vec2 offset = iResolution / vec2(dimensions.y, dimensions.x);

    if (mod(gl_FragCoord.x, offset.x) <= thickness || mod(gl_FragCoord.y, offset.y) <= thickness) {
        gl_FragColor = vec4(1, 1, 1, 1);
    }
}

uniform vec3 FillColor;

uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix
uniform mat4   u_projection;    // Projection matrix

in vec2 position;
in float density;

out vec4 color;

void main()
{
    //gl_Position = vec4(position, 0.0, 1.0);
    gl_Position = u_projection * u_view * u_model * vec4(position, 0.0, 1.0);
    color = vec4(FillColor, density);
}

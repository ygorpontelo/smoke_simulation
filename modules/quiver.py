from glumpy import gl, gloo


vertex      = 'shaders/quiver/quiver.vert'
fragment    = 'shaders/quiver/quiver.frag'


class Quiver:

    def __init__(self, side_count, velocities, width, height) -> None:
        self.dx = 1
        self.dy = 1

        self.program = gloo.Program(vertex, fragment, count=4)

        self.program['position'] = (
            (-self.dx,-self.dy), 
            (-self.dx,+self.dy), 
            (+self.dx,-self.dy), 
            (+self.dx,+self.dy),
        )

        self.program["dim_x"] = side_count
        self.program["dim_y"] = side_count
        self.program["linewidth"] = 1.0
        self.program["iResolution"] = width, height

        self.update_velocities(velocities)

    def draw(self):
        self.program.draw(gl.GL_TRIANGLE_STRIP)
    
    def update_velocities(self, velocities):
        # send without ghost cells
        # v = velocities.to_numpy()[1:-1, 1:-1]
        v = velocities[1:-1, 1:-1]
        self.program["velocities"] = v.view(gloo.TextureFloat2D)

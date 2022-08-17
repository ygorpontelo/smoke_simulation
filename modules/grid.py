from glumpy import gl, gloo


vertex      = 'shaders/grid/grid.vert'
fragment    = 'shaders/grid/grid.frag'


class Grid:

    def __init__(self, side_count, width, height) -> None:
        self.dx = 1
        self.dy = 1

        self.program = gloo.Program(vertex, fragment, count=4)

        self.program['position'] = (
            (-self.dx,-self.dy), 
            (-self.dx,+self.dy), 
            (+self.dx,-self.dy), 
            (+self.dx,+self.dy),
        )

        self.program["thickness"] = 1
        self.program["dimensions"] = side_count, side_count
        self.program["iResolution"] = width, height

    def draw(self):
        self.program.draw(gl.GL_TRIANGLE_STRIP)

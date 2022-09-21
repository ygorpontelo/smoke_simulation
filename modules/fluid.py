from glumpy import gl, gloo
import numpy as np

from modules.grid import Grid
from modules.quiver import Quiver
from modules import solvers


vertex      = 'shaders/fluid/fluid.vert'
fragment    = 'shaders/fluid/fluid.frag'


class Fluid:

    def __init__(self, width, height, cell_count) -> None:
        self.cell_count = cell_count
        self.width = width
        self.height = height

        self.dx = width/cell_count
        self.dy = height/cell_count

        # Ghost cells are used, so each dimension is increased by 2
        self.velocity_field = np.zeros(shape=(cell_count+2, cell_count+2, 2), dtype=np.float32)

        # density field of smoke
        self.density_field  = np.zeros(shape=(cell_count+2, cell_count+2), dtype=np.float32)

        # external forces acting on velocity field
        self.external_forces = np.zeros(shape=(2,), dtype=np.float32)
        self.external_forces[0] = 0
        self.external_forces[1] = 0

        self.grid = Grid(cell_count, width, height) 
        self.vectors = Quiver(cell_count, self.velocity_field, width, height)

        self.show_grid = False
        self.show_vectors = False
        self.smoke_color = 1., 1., 1.
    
        self.program = gloo.Program(vertex, fragment, count=4)
        
        self.program["position"] = self.calculate_vertex_field()

        self.update_smoke_color()
        self.update_density()

    def draw(self):
        # draw smoke first
        self.program.draw(gl.GL_TRIANGLE_STRIP)

        # controls after
        if self.show_grid:
            self.grid.draw()

        if self.show_vectors:
            self.vectors.draw()
    
    def update_smoke_color(self):
        self.program["FillColor"] = self.smoke_color

    def update_density(self):
        self.program["density"] = self.density_field.view(gloo.TextureFloat2D)

    def update_fields(self):
        self.vectors.update_velocities(self.velocity_field)
        self.update_density()
    
    def calculate_vertex_field(self):

        def convert_pos(pos, max_value):
            return (2*pos-max_value)/max_value
        
        vertexes = np.zeros(shape=(self.cell_count+2, self.cell_count+2))
        
        for i in range(self.cell_count):
            for j in range(self.cell_count):
                # pos = 0 -> -1
                # pos = 1/2*width -> 0
                # pos = width -> 1
                # (2*pos-width)/width
                # same for height
                pos = j*self.dx + self.dx/2, self.height - (i*self.dy + self.dy/2)
                pos = convert_pos(pos[0], self.width), convert_pos(pos[1], self.height)
                vertexes[i+1, j+1] = pos
            vertexes[i+1, 0] = (-1, vertexes[i+1, 1][1])
            vertexes[i+1, -1] = (1, vertexes[i+1, -2][1])

        for i in range(self.cell_count+2):
            ...
        

    def solve_fields(self, dt):
        solvers.solve_fields(
            dt, 
            self.dx, 
            self.dy, 
            self.width, 
            self.height, 
            self.density_field, 
            self.velocity_field
        )
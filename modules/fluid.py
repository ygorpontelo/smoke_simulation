from glumpy import gl, gloo, glm
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

        # ghost cells are used, so each dimension is increased by 2
        self.velocity_field = np.zeros(shape=(cell_count+2, cell_count+2, 2), dtype=np.float32)

        # density field of smoke
        self.density_field  = np.zeros(shape=(cell_count+2, cell_count+2), dtype=np.float32)

        # external forces acting on velocity field
        # our case is primary the mouse, so no initial forces
        self.external_forces = np.zeros(shape=(2,), dtype=np.float32)
        self.external_forces[0] = 0
        self.external_forces[1] = 0

        # grid and Vectors representations
        self.grid = Grid(cell_count, width, height) 
        self.vectors = Quiver(cell_count, self.velocity_field, width, height)

        # parameters on the gui
        self.show_grid = False
        self.show_vectors = False
        self.smoke_color = 1., 1., 1.
    
        # opengl program for the smoke
        # count is number of vertexes
        # need to specify glsl version
        self.program = gloo.Program(vertex, fragment, count=(cell_count+2)**2, version="430")
        
        # set vertex coords
        self.program["position"] = self.calculate_vertex_field().view(gloo.VertexBuffer)

        # set index on coords
        # this tells opengl how to draw the triangles
        self.index_buffer = self.calculate_index().view(gloo.IndexBuffer)

        # config initial spacial view
        view = np.eye(4)
        glm.translate(view, 0, 0, -2.4)
        self.view_matrix = [0, 0,-2.4]
        self.program['u_model'] = np.eye(4)
        self.program['u_view'] = view
        self.program['u_projection'] = glm.perspective(45.0, 1, 2.0, 100.0)

        # initial value update
        self.update_smoke_color()
        self.update_density()

    def draw(self):
        # draw smoke first
        self.program.draw(gl.GL_TRIANGLES, self.index_buffer)

        # controls after
        if self.show_grid:
            self.grid.draw()

        if self.show_vectors:
            self.vectors.draw()
    
    def update_smoke_color(self):
        self.program["FillColor"] = self.smoke_color

    def update_density(self):
        self.program["density"] = self.density_field.flatten()

    def update_fields(self):
        """Update density and velocity field values"""

        self.vectors.update_velocities(self.velocity_field)
        self.update_density()
    
    def update_view_matrix(self):
        self.program["u_view"] = glm.translate(
            np.eye(4), self.view_matrix[0], self.view_matrix[1], self.view_matrix[2]
        )

    def solve_fields(self, dt):
        """Call solver for the fields"""

        solvers.solve_fields(
            dt, 
            self.dx, 
            self.dy, 
            self.width, 
            self.height, 
            self.density_field, 
            self.velocity_field
        )

    def calculate_vertex_field(self):
        """Generate vertex coords for smoke field"""

        # convert coord to normalized
        def convert_pos(pos, max_value):
            # pos = 0 -> -1
            # pos = 1/2*width -> 0
            # pos = width -> 1
            # (2*pos-width)/width
            # same for height

            return (2*pos-max_value)/max_value
        
        # coord matrix
        vertexes = np.zeros(shape=(self.cell_count+2, self.cell_count+2, 2))
        
        # calculate coords, ghost cells can be included
        for i in range(-1, self.cell_count + 1):
            for j in range(-1, self.cell_count + 1):
                # calculate position of vertex and normalize
                pos = j*self.dx + self.dx/2, self.height - (i*self.dy + self.dy/2)
                pos = convert_pos(pos[0], self.width), convert_pos(pos[1], self.height)
                vertexes[i+1, j+1] = pos

        # transform matrix to array of coords
        return np.array([vert for row in vertexes for vert in row])

    def calculate_index(self):
        """Generate array with index information"""

        size = self.cell_count+2
        indices = np.zeros((size-1)**2*6, dtype=np.uint32)
        index = 0
        for x in range(size-1):
            for z in range(size-1):
                offset = x*size + z
                indices[index]      = (offset + 0)
                indices[index + 1]  = (offset + 1)
                indices[index + 2]  = (offset + size)
                indices[index + 3]  = (offset + 1)
                indices[index + 4]  = (offset + size + 1)
                indices[index + 5]  = (offset + size)
                index += 6

        return indices


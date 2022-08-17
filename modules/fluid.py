from glumpy import gl, gloo

import taichi as ti

from modules.grid import Grid
from modules.quiver import Quiver


vertex      = 'shaders/fluid/fluid.vert'
fragment    = 'shaders/fluid/fluid.frag'


@ti.data_oriented
class Fluid:

    def __init__(self, width, height, cell_count) -> None:
        self.cell_count = cell_count
        self.width = width
        self.height = height

        self.dx = width/cell_count
        self.dy = height/cell_count

        # Ghost cells are used, so each dimension is increased by 2
        self.velocity_field = ti.Vector.field(dtype=ti.f32, n=2, shape=(cell_count+2, cell_count+2))

        # density field of smoke, has resolution of the screen        
        self.density_field  = ti.field(ti.f32, shape=(cell_count+2, cell_count+2))

        # external forces acting on velocity field
        self.external_forces = ti.field(ti.f32, shape=(2,))
        self.external_forces[0] = 0
        self.external_forces[1] = 0.1

        self.grid = Grid(cell_count, width, height) 
        self.vectors = Quiver(cell_count, self.velocity_field, width, height)

        self.show_grid = False
        self.show_vectors = False
        self.smoke_color = 1., 1., 1.
    
        self.program = gloo.Program(vertex, fragment, count=4)
        
        self.program["position"] = (
            (-1, -1), 
            (-1, +1), 
            (+1, -1), 
            (+1, +1),
        )

        self.program["scale"] =  1.0/width, 1.0/height

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
        self.program["density"] = self.density_field.to_numpy().view(gloo.TextureFloat2D)

    def update_fields(self):
        self.vectors.update_velocities(self.velocity_field)
        self.update_density()

    # ---------- Taichi context lvl funcs ----------
    @ti.kernel
    def solve_fields(self, dt: ti.f32):
        self.vel_step(dt)
        self.dens_step(dt)
    
    @ti.func
    def dens_step(self, dt: ti.f32):
        self.difuse_step(dt)
        self.advect(dt, self.density_field)
    
    @ti.func
    def vel_step(self, dt: ti.f32):
        # self.add_forces(dt)
        self.difuse_vel_step(dt)
        self.advect_vel(dt, self.velocity_field)
        self.project(self.velocity_field)
    
    @ti.func
    def update_bnd(self, value_field: ti.template()):
        # slicing doesn't work on fields
        s = value_field.shape
        
        # rows
        for i in range(s[1]):
            value_field[0, i] = value_field[1, i]
            value_field[s[0]-1, i] = value_field[s[0]-2, i]
        
        # cols
        for i in range(s[0]):
            value_field[i, 0] = value_field[i, 1]
            value_field[i, s[1]-1] = value_field[i, s[1]-2]

    @ti.func
    def update_bnd_vel(self, value_field: ti.template()):
        # slicing doesn't work on fields
        s = value_field.shape
        
        # rows
        for i in range(s[1]):
            value_field[0, i] = [value_field[1, i][0], -value_field[1, i][1]]
            value_field[s[0]-1, i] = [value_field[s[0]-2, i][0], -value_field[s[0]-2, i][1]]
        
        # cols
        for i in range(s[0]):
            value_field[i, 0] = [-value_field[i, 1][0], value_field[1, i][1]]
            value_field[i, s[1]-1] = [-value_field[i, s[1]-2][0], value_field[i, s[1]-2][1]]

    # density funcs
    @ti.func
    def difuse(self, dt: ti.f32, x0):
        # density field x0 is passed by value not ref
        s = self.density_field.shape
        a = dt

        for i in range(1, s[0]-1):
            for j in range(1, s[1]-1):
                v = (
                    x0[i, j] + a *
                    (
                        (x0[i-1, j] + x0[i+1, j] + x0[i, j-1] + x0[i, j+1])/(4.0)
                    )
                ) / (1+a)
                self.density_field[i, j] = v

    @ti.func
    def difuse_step(self, dt: ti.f32):
        # solve system with 20 iterations
        for _ in range(1):
            for it in range(20):
                self.difuse(dt, self.density_field)
                self.update_bnd(self.density_field)

    @ti.func
    def advect(self, dt: ti.f32, d0):
        s = self.density_field.shape

        for i in range(1, s[0]-1):
            for j in range(1, s[1]-1):
                vel = self.velocity_field[i, j]
            
                # pos back in time
                # i and j are inverted for spacial coordinates
                x = (j*self.dx+self.dx/2) - dt*vel[0]
                y = (i*self.dy+self.dy/2) - dt*vel[1]

                if x < 0:
                    x = 0
                if x > self.width:
                    x = self.width
                if y < 0:
                    y = 0
                if y > self.height:
                    y = self.height
                
                sqX = (x-self.dx/2)/self.dx
                sqY = (y-self.dy/2)/self.dy

                # indices of the four cells back in time
                i0 = int(sqY)
                j0 = int(sqX)

                kx = sqX - int(sqX)
                ky = sqY - int(sqY)
                
                if i0 < 0:
                    i0 = 0
                if i0 > s[0]-2:
                    i0 = s[0]-2
                if j0 < 0:
                    j0 = 0
                if j0 > s[1]-2:
                    j0 = s[1]-2

                z1 = (1 - kx) * d0[i0, j0] + kx * d0[i0, j0+1]
                z2 = (1 - kx) * d0[i0+1, j0] + kx * d0[i0+1, j0+1]
                v = (1 - ky) * z1 + ky * z2

                self.density_field[i, j] = v
    
        self.update_bnd(self.density_field)
    
    # velocity funcs
    @ti.func
    def add_forces(self, dt: ti.f32):
        for i, j in self.velocity_field:
            self.velocity_field[i, j][0] = self.external_forces[0] * dt
            self.velocity_field[i, j][1] = self.external_forces[1] * dt

    @ti.func
    def difuse_vel(self, dt, x0):
        s = self.velocity_field.shape
        a = dt

        for i in range(1, s[0]-1):
            for j in range(1, s[1]-1):
                v = (
                    x0[i, j] + a *
                    (
                        (x0[i-1, j] + x0[i+1, j] + x0[i, j-1] + x0[i, j+1])/(4.0)
                    )
                ) / (1+a)
                self.velocity_field[i, j] = v
    
    @ti.func
    def difuse_vel_step(self, dt):
        # solve system with 20 iterations
        for _ in range(1):
            for it in range(20):
                self.difuse_vel(dt, self.velocity_field)
                self.update_bnd_vel(self.velocity_field)
    
    @ti.func
    def advect_vel(self, dt, d0):
        s = self.velocity_field.shape

        for i in range(1, s[0]-1):
            for j in range(1, s[1]-1):
                vel = self.velocity_field[i, j]

                # pos back in time
                # i and j are inverted for spacial coordinates
                x = (j*self.dx+self.dx/2) - dt*vel[0]
                y = (i*self.dy+self.dy/2) - dt*vel[1]

                if x < 0:
                    x = 0
                if x > self.width:
                    x = self.width
                if y < 0:
                    y = 0
                if y > self.height:
                    y = self.height

                sqX = (x-self.dx/2)/self.dx
                sqY = (y-self.dy/2)/self.dy
                
                # indices of the four cells back in time
                i0 = int(sqY)
                j0 = int(sqX)

                kx = sqX - int(sqX)
                ky = sqY - int(sqY)
                
                if i0 < 0:
                    i0 = 0
                if i0 > s[0]-2:
                    i0 = s[0]-2
                if j0 < 0:
                    j0 = 0
                if j0 > s[1]-2:
                    j0 = s[1]-2

                z1 = (1 - kx) * d0[i0, j0] + kx * d0[i0, j0+1]
                z2 = (1 - kx) * d0[i0+1, j0] + kx * d0[i0+1, j0+1]
                v = (1 - ky) * z1 + ky * z2
                
                self.velocity_field[i, j] = v
                
        self.update_bnd_vel(self.velocity_field)

    @ti.func
    def project(self, prev_vel):
        s = self.velocity_field.shape

        for i in range(1, s[0]-1):
            for j in range(1, s[1]-1):
                prev_vel[i, j][1] = ( 
                    # divergence
                    (
                        (self.velocity_field[i, j+1][0] - self.velocity_field[i, j-1][0])/(-2*self.dx) + 
                        (self.velocity_field[i+1, j][1] - self.velocity_field[i-1, j][1])/(-2*self.dy)
                    )
                )
                prev_vel[i, j][0] = 0.0
        self.update_bnd(prev_vel)

        # avoid parallel for in this case
        for _ in range(1):
            for it in range(20):
                for i in range(1, s[0]-1):  
                    for j in range(1, s[1]-1):
                        prev_vel[i, j][0] = ( 
                            prev_vel[i-1, j][0] + 
                            prev_vel[i+1, j][0] + 
                            prev_vel[i, j-1][0] + 
                            prev_vel[i, j+1][0] +
                            prev_vel[i, j][1]) / (4.0)
                self.update_bnd(prev_vel)

        for i in range(1, s[0]-1):
            for j in range(1, s[1]-1):
                # gradient
                self.velocity_field[i, j][0] -= (prev_vel[i, j+1][0] - prev_vel[i, j-1][0])/(2*self.dx)
                self.velocity_field[i, j][1] -= (prev_vel[i+1, j][0] - prev_vel[i-1, j][0])/(2*self.dy)
        self.update_bnd_vel(self.velocity_field)

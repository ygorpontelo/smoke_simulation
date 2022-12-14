"""
Implementation of functions to solve the fields.

Using numba for performance.
"""

from numba import njit, prange


# number of solver iterations
n_iter = 25

# which solver
solver_gauss = False


##### Exposed funcs #####
def solve_fields(dt, dx, dy, width, height, density_field, velocity_field):
    vel_step(dt, dx, dy, width, height, velocity_field)
    dens_step(dt, dx, dy, width, height, density_field, velocity_field)


def dens_step(dt, dx, dy, width, height, density_field, velocity_field):
    difuse_step(dt, density_field)
    advect(dt, dx, dy, width, height, density_field, velocity_field)


def vel_step(dt, dx, dy, width, height, velocity_field):
    # add forces is the mouse in our case
    # self.add_forces(dt)
    # two projections increase stability
    difuse_vel_step(dt, velocity_field)
    project(dx, dy, velocity_field)
    advect_vel(dt, dx, dy, width, height, velocity_field)
    project(dx, dy, velocity_field)


##### Boundaries funcs #####
@njit
def update_bnd(original_field):
    s = original_field.shape
    value_field = original_field.copy()
    
    # rows
    original_field[0, :] = value_field[1, :]
    original_field[s[0]-1, :] = value_field[s[0]-2, :]
    
    # cols
    original_field[:, 0] = value_field[:, 1]
    original_field[:, s[1]-1] = value_field[:, s[1]-2]

@njit(parallel=True)
def update_bnd_vel(original_field):
    s = original_field.shape
    value_field = original_field.copy()
    
    # rows
    for i in prange(s[1]):
        original_field[0, i] = [value_field[1, i, 0], -value_field[1, i, 1]]
        original_field[s[0]-1, i] = [value_field[s[0]-2, i, 0], -value_field[s[0]-2, i, 1]]
    
    # cols
    for i in prange(s[0]):
        original_field[i, 0] = [-value_field[i, 1, 0], value_field[i, 1, 1]]
        original_field[i, s[1]-1] = [-value_field[i, s[1]-2, 0], value_field[i, s[1]-2, 1]]


###### density funcs #####
def difuse_step(dt, density_field):
    # solve system with n iterations
    if solver_gauss:
        gauss_siedel(density_field, dt, 2.0)
    else:
        jacobi(density_field, dt, 2.0)

@njit(parallel=True)
def advect(dt, dx, dy, width, height, density_field, velocity_field):
    s = density_field.shape
    d0 = density_field.copy()

    for i in prange(1, s[0]-1):
        for j in range(1, s[1]-1):
            vel = velocity_field[i, j]
        
            # pos back in time
            # i and j are inverted for spacial coordinates
            x = (j*dx + dx/2) - dt*vel[0]
            y = (i*dy + dy/2) - dt*vel[1]

            if x < 0:
                x = 0
            if x > width:
                x = width
            if y < 0:
                y = 0
            if y > height:
                y = height
            
            sqX = (x-dx/2)/dx
            sqY = (y-dy/2)/dy

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

            density_field[i, j] = (1 - ky) * z1 + ky * z2

    update_bnd(density_field)


##### velocity funcs #####
def difuse_vel_step(dt, velocity_field):
    # solve system with n iterations
    if solver_gauss:
        gauss_siedel(velocity_field, dt)
    else:
        jacobi(velocity_field, dt)

@njit(parallel=True)
def advect_vel(dt, dx, dy, width, height, velocity_field):
    s = velocity_field.shape
    d0 = velocity_field.copy()

    for i in prange(1, s[0]-1):
        for j in range(1, s[1]-1):
            vel = velocity_field[i, j]

            # pos back in time
            # i and j are inverted for spacial coordinates
            x = (j*dx + dx/2) - dt*vel[0]
            y = (i*dy + dy/2) - dt*vel[1]

            if x < 0:
                x = 0
            if x > width:
                x = width
            if y < 0:
                y = 0
            if y > height:
                y = height

            sqX = (x-dx/2)/dx
            sqY = (y-dy/2)/dy
            
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
            
            velocity_field[i, j] = (1 - ky) * z1 + ky * z2
            
    update_bnd_vel(velocity_field)

@njit(parallel=True)
def project(dx, dy, velocity_field):
    s = velocity_field.shape
    prev_vel = velocity_field.copy()

    for i in prange(1, s[0]-1):
        for j in range(1, s[1]-1):
            prev_vel[i, j][1] = ( 
                # divergence
                (velocity_field[i, j+1, 0] - velocity_field[i, j-1, 0]) / (-2.0*dx) + 
                (velocity_field[i+1, j, 1] - velocity_field[i-1, j, 1]) / (-2.0*dy)
            )
            prev_vel[i, j][0] = 0
    update_bnd(prev_vel)
    
    # solve div system
    if solver_gauss:
        gauss_siedel_project(prev_vel)
    else:
        jacobi_project(prev_vel)

    for i in prange(1, s[0]-1):
        for j in range(1, s[1]-1):
            # gradient
            velocity_field[i, j][0] -= (prev_vel[i, j+1][0] - prev_vel[i, j-1][0]) / (2.0*dx)
            velocity_field[i, j][1] -= (prev_vel[i+1, j][0] - prev_vel[i-1, j][0]) / (2.0*dy)
    update_bnd_vel(velocity_field)


##### Solvers #####
@njit(parallel=True)
def gauss_siedel(field_vector, dt, a_mod = 1.0):
    s = field_vector.shape
    a = dt/n_iter * a_mod
    for it in range(n_iter):
        x0 = field_vector.copy()

        for i in prange(1, s[0]-1):
            for j in range(1, s[1]-1):
                field_vector[i, j] = (
                    x0[i, j] + a *
                    (
                        (field_vector[i-1, j] + field_vector[i+1, j] + field_vector[i, j-1] + field_vector[i, j+1])/(4.0)
                    )
                ) / (1+a)
        update_bnd(field_vector)

@njit(parallel=True)
def gauss_siedel_project(field_vector):
    s = field_vector.shape
    for it in range(n_iter):
        value = field_vector.copy()
        for i in prange(1, s[0]-1):  
            for j in range(1, s[1]-1):
                field_vector[i, j][0] = ( 
                    field_vector[i-1, j][0] + 
                    field_vector[i+1, j][0] + 
                    field_vector[i, j-1][0] + 
                    field_vector[i, j+1][0] +
                    value[i, j][1]
                ) / 4.0
        update_bnd(field_vector)

@njit(parallel=True)
def jacobi(field_vector, dt, a_mod = 1.0):
    s = field_vector.shape
    a = dt/n_iter * a_mod
    for it in range(n_iter):
        x0 = field_vector.copy()

        for i in prange(1, s[0]-1):
            for j in range(1, s[1]-1):
                field_vector[i, j] = (
                    x0[i, j] + a *
                    (
                        (x0[i-1, j] + x0[i+1, j] + x0[i, j-1] + x0[i, j+1])/(4.0)
                    )
                ) / (1+a)
        update_bnd(field_vector)

@njit(parallel=True)
def jacobi_project(field_vector):
    s = field_vector.shape
    for it in range(n_iter):
        value = field_vector.copy()
        for i in prange(1, s[0]-1):  
            for j in range(1, s[1]-1):
                field_vector[i, j][0] = ( 
                    value[i-1, j][0] + 
                    value[i+1, j][0] + 
                    value[i, j-1][0] + 
                    value[i, j+1][0] +
                    value[i, j][1]
                ) / 4.0
        update_bnd(field_vector)

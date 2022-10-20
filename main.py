import pyglet

import imgui
from imgui.integrations.pyglet import PygletProgrammablePipelineRenderer

from glumpy import app

# our modules
from modules import fluid


# Use pyglet as backend
# using opengl 4.3
app.use("pyglet", major=4, minor=3)

# Constants
WIDTH = 900
HEIGHT = 900
CELLS = 128

# create window with openGL context
window = app.Window(WIDTH, HEIGHT)

# create renderer of imgui on window
imgui.create_context()
imgui_renderer = PygletProgrammablePipelineRenderer(window.native_window) # pass native pyglet window.

fps_display = pyglet.window.FPSDisplay(window=window.native_window)

# main object
smoke_grid = fluid.Fluid(WIDTH, HEIGHT, CELLS)

# draw only lines, no rasterization, good for tests
# gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)


@window.event
def on_draw(dt):
    window.clear()

    fps_display.draw()

    smoke_grid.update_fields()
    smoke_grid.solve_fields(dt)

    # draw smoke first
    smoke_grid.draw()

    # Imgui Interface
    imgui.new_frame()

    imgui.begin("Controls")
    _, smoke_grid.show_grid = imgui.checkbox("Show Grid", smoke_grid.show_grid)
    _, smoke_grid.show_vectors = imgui.checkbox("Show Vectors", smoke_grid.show_vectors)

    changed, smoke_grid.smoke_color = imgui.color_edit3("Smoke Color", *smoke_grid.smoke_color)

    if changed:
        smoke_grid.update_smoke_color()

    changed,  vm = imgui.drag_float3("View Matrix", *smoke_grid.view_matrix, change_speed=0.01)
    smoke_grid.view_matrix = list(vm)
    if changed:
        smoke_grid.update_view_matrix()

    imgui.end()

    # render gui on top of everything
    try:
        imgui.render()
        imgui_renderer.render(imgui.get_draw_data())
    except Exception:
        imgui_renderer.shutdown()

@window.event
def on_mouse_drag(x, y, dx, dy, buttons):
    """The mouse was moved with some buttons pressed."""

    # Case was right mouse button
    if buttons == 4:
        radius = 3
        
        # will be inverted in opengl
        idrow = CELLS - (int(y/smoke_grid.dx) + 1)
        idcol = int(x/smoke_grid.dy) + 1

        if idrow-radius < 0:
            idrow = radius
        elif idrow+radius > CELLS-1:
            idrow = CELLS-1-radius
        
        if idcol-radius < 0:
            idcol = radius
        elif idcol+radius > CELLS-1:
            idcol = CELLS-1-radius
        
        for i in range(-radius, radius):
            idx = idrow + i
            for j in range(-radius, radius):
                idy = idcol + j
                smoke_grid.density_field[idx, idy] = 1

    # Case was left mouse button
    if buttons == 1:
        radius = 2
    
        idrow = CELLS - (int(y/smoke_grid.dx))
        idcol = int(x/smoke_grid.dy)

        if idrow-radius < 0:
            idrow = radius
        elif idrow+radius > CELLS-1:
            idrow = CELLS-1-radius
        
        if idcol-radius < 0:
            idcol = radius
        elif idcol+radius > CELLS-1:
            idcol = CELLS-1-radius

        for i in range(-radius, radius):
            idx = idrow + i
            for j in range(-radius, radius):
                idy = idcol + j
                speed = 1000
                smoke_grid.velocity_field[idx, idy] += [
                    speed*dx, speed*dy
                ]

@window.event
def on_show():
    # disable resize on show
    window.native_window.set_minimum_size(WIDTH, HEIGHT)
    window.native_window.set_maximum_size(WIDTH, HEIGHT)


@window.event    
def on_mouse_scroll(x, y, dx, dy):
    'The mouse wheel was scrolled by (dx,dy).'
    smoke_grid.view_matrix[-1] -= dy*0.1   
    smoke_grid.update_view_matrix()


if __name__ == "__main__":
    # run app
    app.run()

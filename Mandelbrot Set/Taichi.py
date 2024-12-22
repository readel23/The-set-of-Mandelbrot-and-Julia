import pygame
import taichi as ti

ti.init(arch=ti.gpu)

width, height = 1200, 1200
screen = pygame.display.set_mode((width, height))

x_min, x_max = -2.0, 2.0
y_min, y_max = -2.0, 2.0
max_iter = 1000

pixels = ti.Vector.field(3, dtype=ti.u8, shape=(width, height))

@ti.kernel
def mandelbrot(x_min: float, x_max: float, y_min: float, y_max: float, max_iter: int):
    for i, j in pixels:
        x = x_min + i / width * (x_max - x_min)
        y = y_min + j / height * (y_max - y_min)
        z_real, z_imag = x, y
        iteration = 0

        while z_real * z_real + z_imag * z_imag <= 4 and iteration < max_iter:
            z_real, z_imag = (
                z_real * z_real - z_imag * z_imag + x,
                2.0 * z_real * z_imag + y,
            )
            iteration += 1

        pixels[i, j] = ti.Vector([
            iteration * 3 % 255,
            iteration * 12 % 255,
            iteration * 7 % 255,
        ])

def render_mandelbrot():
    surface = pygame.surfarray.make_surface(pixels.to_numpy())
    screen.blit(surface, (0, 0))

clock = pygame.time.Clock()
RUN = True
while RUN:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            RUN = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                zoom_factor = 0.5
            elif event.button == 3:
                zoom_factor = 2.0

            mouse_x, mouse_y = pygame.mouse.get_pos()
            c_real = x_min + mouse_x / width * (x_max - x_min)
            c_imag = y_min + mouse_y / height * (y_max - y_min)

            range_x = (x_max - x_min) * zoom_factor
            range_y = (y_max - y_min) * zoom_factor

            x_min = c_real - range_x / 2
            x_max = c_real + range_x / 2
            y_min = c_imag - range_y / 2
            y_max = c_imag + range_y / 2

    mandelbrot(x_min, x_max, y_min, y_max, max_iter)
    render_mandelbrot()

    pygame.display.set_caption(f"The Mandelbrot Set with Taichi | FPS {clock.get_fps():.2f}")
    clock.tick(240)
    pygame.display.update()

pygame.quit()

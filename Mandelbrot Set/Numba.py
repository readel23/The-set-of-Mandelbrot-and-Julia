import pygame
import numpy as np
import numba

pygame.init()
width, height = 1200, 1200

screen = pygame.display.set_mode((width, height))

max_iter = 1000
x_min, x_max = -2.0, 2.0
y_min, y_max = -2.0, 2.0
x = np.linspace(x_min, x_max, width)
y = np.linspace(y_min, y_max, height)

@numba.njit(fastmath=True, parallel=True)
def mandelbrot(x, y, width, height, max_iter):
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    for i in numba.prange(width):
        for j in range(height):
            c = x[i] + 1j * y[j]
            z = 0
            iteration = 0
            while abs(z) <= 2 and iteration < max_iter:
                z = z**2 + c
                iteration += 1
            color = (iteration * 3 % 255, iteration * 12 % 255, iteration * 7 % 255)
            pixels[i, j] = color
    return pixels

def render_mandelbrot():
    pixels = mandelbrot(x, y, width, height, max_iter)
    pygame.surfarray.blit_array(screen, pixels)

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
            c_real = x[mouse_x]
            c_imag = y[mouse_y]
            range_x = (x_max - x_min) * zoom_factor
            range_y = (y_max - y_min) * zoom_factor

            x_min = c_real - range_x / 2
            x_max = c_real + range_x / 2
            y_min = c_imag - range_y / 2
            y_max = c_imag + range_y / 2

            x = np.linspace(x_min, x_max, width)
            y = np.linspace(y_min, y_max, height)

        render_mandelbrot()
        pygame.display.set_caption(f"The Mandelbrot Set with Numba | FPS {clock.get_fps():.2f}")
        clock.tick(240)
        pygame.display.update()

pygame.quit()
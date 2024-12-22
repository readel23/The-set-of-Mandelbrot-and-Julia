import pygame
import numpy as np

pygame.init()
width, height = 1200, 1200

screen = pygame.display.set_mode((width, height))

max_iter = 100
x_min, x_max = -2.0, 2.0
y_min, y_max = -2.0, 2.0
x = np.linspace(x_min, x_max, width)
y = np.linspace(y_min, y_max, height)

u, v = 428, 428
c = x[u] + 1j * y[v]

for i in range(width):
    for j in range(height):
        z = x[i] + 1j * y[j]
        iteration = 0
        while abs(z) <= 2 and iteration < max_iter:
            z = z**2 + c
            iteration += 1
        color = (iteration * 3 % 255, iteration * 12 % 255, iteration * 7 % 255)
        screen.set_at((i, j), color)

clock = pygame.time.Clock()
RUN = True
while RUN:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            RUN = False
    pygame.display.set_caption(f"The Julia Set | FPS: {clock.get_fps():.2f}")
    clock.tick(240)
    pygame.display.update()

pygame.quit()
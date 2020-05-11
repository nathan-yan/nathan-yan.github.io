import numpy as np
import math

state =  np.random.uniform(low=-0.05, high=0.05, size=(4,))

gravity = 9.8
masspole = 0.1
masscart = 1.0
length = 0.5
total_mass = (masspole + masscart)
polemass_length = masspole * length
force_mag = 10.0
tau = 0.02
x_threshold = 2.5
theta_threshold_radians = 12 * 2 * math.pi / 360.

def step(state, action):
    x, x_dot, theta, theta_dot = state
    force = force_mag if action == 1 else -force_mag
    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    # For the interested reader:
    # https://coneural.org/florian/papers/05_cart_pole.pdf
    temp = (force + polemass_length * theta_dot ** 2 * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    x = x + tau * x_dot
    x_dot = x_dot + tau * xacc
    theta = theta + tau * theta_dot
    theta_dot = theta_dot + tau * thetaacc


    state = (x, x_dot, theta, theta_dot)

    done = bool(
        x < -x_threshold
        or x > x_threshold
        or theta < -theta_threshold_radians
        or theta > theta_threshold_radians
    )

    return state, done

import pygame
from pygame.locals import *
pygame.init()
import time

done = False
screen = pygame.display.set_mode((500, 500))

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            done = True
    
    state, _ = step(state, np.random.randint(0, 2))
    x_pos, _, theta, _ = state

    theta = theta + np.pi / 2

    screen.fill((255, 255, 255))
    pygame.draw.line(screen, (0, 0, 0), (int(x_pos * 100) + 250, 500), (int(x_pos * 100 + np.cos(theta) * 100) + 250, int(500 - np.sin(theta) * 100)), 2)
    pygame.display.flip()

    time.sleep(0.01)
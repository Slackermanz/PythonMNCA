import pygame
import sys
from world import World

# Initialize Pygame
pygame.init()

# Set the size of the window and scale factor
window_size = (512, 512)
scale_factor = 6  # Scale by 2^n, for n=3

# Create the window
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Pygame with World")

# Create a World instance with a scale factor
world = World(window_size, scale=scale_factor)

# Seed the world with random values
world.seed()

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update the world
    world.step()

    # Render the world state to the screen
    world.render(screen)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()

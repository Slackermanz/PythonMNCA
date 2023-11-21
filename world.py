import numpy as np
import pygame
from convolve import MNCA_Test

class World:
    def __init__(self, size, scale=1, dtype=np.float32, channels=1):
        self.size = size
        self.scale = scale  # The scale factor
        self.dtype = dtype
        self.channels = channels

        # Adjust buffer shape for scale
        scaled_size = (size[0] // scale, size[1] // scale)
        buffer_shape = scaled_size if channels == 1 else (*scaled_size, channels)
        
        self.buffer1 = np.zeros(buffer_shape, dtype=dtype)
        self.buffer2 = np.zeros(buffer_shape, dtype=dtype)

    def seed(self):
        if self.channels == 1:
            # Greyscale seeding
            if np.issubdtype(self.dtype, np.integer):
                self.buffer1 = np.random.randint(0, 256, self.size, dtype=self.dtype)
            elif np.issubdtype(self.dtype, np.floating):
                self.buffer1 = np.random.random(self.size).astype(self.dtype)
            elif np.issubdtype(self.dtype, np.bool_):
                self.buffer1 = np.random.choice([True, False], self.size)
            else:
                raise ValueError("Unsupported data type for buffer.")
        else:
            # RGB seeding
            if np.issubdtype(self.dtype, np.integer):
                self.buffer1 = np.random.randint(0, 256, (*self.size, self.channels), dtype=self.dtype)
            elif np.issubdtype(self.dtype, np.floating):
                self.buffer1 = np.random.random((*self.size, self.channels)).astype(self.dtype)
            else:
                raise ValueError("Unsupported data type for RGB buffer.")

    def step(self):
        # Perform convolution on buffer1 and store the result in buffer2
        self.buffer2 = MNCA_Test(self.buffer1)

        # Swap the buffer references
        self.buffer1, self.buffer2 = self.buffer2, self.buffer1

    def get_state(self):
        return self.buffer1

    def render(self, screen):
        # Get the current state
        state_data = self.get_state()

        # Handle grayscale or RGB data
        if self.channels == 1:
            # Scale the values to the range 0-255 and expand to 3 channels
            scaled_data = np.clip(state_data, 0.0, 1.0) * 255
            scaled_data = scaled_data.astype(np.uint8)  # Convert to unsigned byte
            rgb_data = np.stack((scaled_data,) * 3, axis=-1)
        else:
            # If RGB, ensure the data is in the correct format (0-255 range)
            rgb_data = np.clip(state_data, 0, 255).astype(np.uint8)

        # Create a Pygame surface from the RGB data
        state_surface = pygame.surfarray.make_surface(rgb_data)

        # Resize the surface to the desired dimensions using Pygame's scale functionality
        if self.scale != 1:
            window_size = (self.size[0] * self.scale, self.size[1] * self.scale)
            state_surface = pygame.transform.scale(state_surface, window_size)

        # Blit the surface onto the screen
        screen.blit(state_surface, (0, 0))

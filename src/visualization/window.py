import pygame
import sys

class Window:
    def __init__(self, width=800, height=600, title="Polygon Packing Visualization"):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Set up colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)

        self.flash = True
        self.color = self.WHITE
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    if self.flash:
                        self.color = self.BLACK
                    else:
                        self.color = self.WHITE
                    self.flash = not self.flash


            
    
    def run(self):
        while self.running:
            # Handle events
            self.handle_events()
            
            # Clear screen
            self.screen.fill(self.color)
            
            # Update display
            pygame.display.flip()
            
            # Control frame rate (60 FPS)
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

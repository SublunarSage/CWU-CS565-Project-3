import pygame
import numpy as np
from typing import List, Tuple
from algorithms.polygon_helpers import compute_polygon_vertices_cpu, check_polygon_collision

def draw_solution(screen: pygame.Surface,
                 boundary: np.ndarray,
                 polygons: List[Tuple[int, float]],
                 solution: np.ndarray,
                 scale: float = 1.0,
                 margin: int = 100):
    """
    Draw the polygon packing solution on the screen.

    Args:
        screen (pygame.Surface): Pygame surface to draw on
        boundary (np.ndarray): Boundary polygon vertices
        polygons (List[Tuple[int, float]]): List of (num_sides, size) tuples
        solution (np.ndarray): Solution array containing positions and rotations
        scale (float, optional): Scale factor for drawing. Defaults to 1.0
        margin (int, optional): Minimum margin from screen edges. Defaults to 100

    Color coding:
        - Green: Valid placement (no collisions)
        - Red: Collision with other polygons
        - Yellow: Collision with boundary
        - Pink: Both types of collisions
    """
    # Calculate boundary box dimensions
    min_x = min(p[0] for p in boundary)
    max_x = max(p[0] for p in boundary)
    min_y = min(p[1] for p in boundary)
    max_y = max(p[1] for p in boundary)
    
    boundary_width = (max_x - min_x) * scale
    boundary_height = (max_y - min_y) * scale
    
    # Calculate centering offsets
    screen_width = screen.get_width()
    screen_height = screen.get_height()
    
    offset_x = (screen_width - boundary_width) / 2
    offset_y = (screen_height - boundary_height) / 2
    
    # Ensure minimum margins are maintained
    offset_x = max(margin, offset_x)
    offset_y = max(margin, offset_y)
    
    # Draw boundary
    scaled_boundary = [(int(x * scale + offset_x), 
                       int(y * scale + offset_y)) for x, y in boundary]
    pygame.draw.polygon(screen, (100, 100, 100), scaled_boundary, 2)
    
    # Pre-compute all polygon vertices for collision checking
    all_vertices = []
    for i, (num_sides, size) in enumerate(polygons):
        base_idx = i * 3
        x = solution[base_idx]
        y = solution[base_idx + 1]
        rotation = solution[base_idx + 2]
        vertices = compute_polygon_vertices_cpu(x, y, rotation, num_sides, size)
        all_vertices.append(vertices)
    
    # Draw packed polygons with collision-based colors
    for i, ((num_sides, size), vertices) in enumerate(zip(polygons, all_vertices)):
        # Check collisions
        boundary_collision = not check_polygon_collision(vertices, boundary, is_boundary=True)
        polygon_collision = False
        
        # Check collision with other polygons
        for j, other_vertices in enumerate(all_vertices):
            if i != j and check_polygon_collision(vertices, other_vertices):
                polygon_collision = True
                break
        
        # Determine color based on collision state
        if polygon_collision and boundary_collision:
            color = (255, 192, 203)  # Pink (both collisions)
        elif polygon_collision:
            color = (255, 0, 0)      # Red (polygon collision)
        elif boundary_collision:
            color = (255, 255, 0)    # Yellow (boundary collision)
        else:
            color = (0, 255, 0)      # Green (no collision)
        
        # Draw the polygon with scaled coordinates
        scaled_vertices = [
            (int(x * scale + offset_x), int(y * scale + offset_y))
            for x, y in vertices
        ]
        pygame.draw.polygon(screen, color, scaled_vertices, 2)
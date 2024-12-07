import numpy as np
import pygame
from algorithms.cuda_polygon_ga import CUDAPolygonPacker
from visualization.polygon_viz import draw_solution
from algorithms.polygon_helpers import compute_polygon_vertices_cpu, check_polygon_collision
from utils.config_loader import load_config


def main():
    boundary, polygons, ga_params, penalties = load_config('config.yaml')

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Polygon Packing Test")
    clock = pygame.time.Clock()
    
    
    # Initialize packer
    packer = CUDAPolygonPacker(
        boundary_polygon=boundary,
        regular_polygons=polygons,
        population_size=ga_params['population_size'],
        mutation_rate=ga_params['mutation_rate'],
        penalties=penalties,
        random_seed=ga_params['random_seed']
    )
    

    # State variables
    evolution_running = False  # Changed from evolution_started to evolution_running
    evolution_started = False  # Keep track if evolution has ever started
    generation = 0
    max_generations = ga_params['max_generations']
    current_best_solution = None
    best_fitness = float('inf')
    
    # Visualization loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                # Modified: Toggle evolution with spacebar
                elif event.key == pygame.K_SPACE:
                    evolution_running = not evolution_running
                    evolution_started = True
                    if evolution_running:
                        print("Evolution resumed! Press SPACE to pause, ESC to quit.")
                    else:
                        print("Evolution paused! Press SPACE to resume, ESC to quit.")
        
        # Run evolution step if active
        if evolution_running and generation < max_generations:
            # Run one generation at a time
            fitness_history, final_population = packer.evolve(generations=1)
            
            # Update best solution and fitness
            generation += 1
            current_fitness = min(fitness_history)
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_idx = np.argmin(fitness_history)
                current_best_solution = final_population[best_idx]

        # Clear screen
        screen.fill((255, 255, 255))
        
        # Display information
        font = pygame.font.Font(None, 36)
        
        # Show different messages based on state
        if not evolution_started:
            # Initial state
            text = "Press SPACE to start evolution"
            font = pygame.font.Font(None, 48)
            text_surface = font.render(text, True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=(screen.get_width()/2, screen.get_height()/2))
            screen.blit(text_surface, text_rect)
        else:
            # Show status, generation, and fitness
            status_text = "PAUSED" if not evolution_running else "RUNNING"
            status_surface = font.render(f"Status: {status_text}", True, (0, 0, 0))
            gen_surface = font.render(f"Generation: {generation}", True, (0, 0, 0))

            # Convert fitness to utilization percentage
            if current_best_solution is not None:
                utilization = packer.get_utilization(current_best_solution)
                fitness_surface = font.render(f"Space Utilization: {utilization:.1f}%", True, (0, 0, 0))
            else:
                fitness_surface = font.render("Space Utilization: 0.0%", True, (0, 0, 0))
            
            screen.blit(status_surface, (10, 10))
            screen.blit(gen_surface, (10, 50))
            screen.blit(fitness_surface, (10, 90))
            
            if generation >= max_generations:
                # Check if solution has any collisions
                has_collisions = False
                
                # Pre-compute all polygon vertices for final check
                all_vertices = []
                for i, (num_sides, size) in enumerate(polygons):
                    base_idx = i * 3
                    x = current_best_solution[base_idx]
                    y = current_best_solution[base_idx + 1]
                    rotation = current_best_solution[base_idx + 2]
                    vertices = compute_polygon_vertices_cpu(x, y, rotation, num_sides, size)
                    all_vertices.append(vertices)
                
                # Check for any collisions
                for i, vertices in enumerate(all_vertices):
                    # Check boundary collision
                    if not check_polygon_collision(vertices, boundary, is_boundary=True):
                        has_collisions = True
                        break
                        
                    # Check collision with other polygons
                    for j, other_vertices in enumerate(all_vertices):
                        if i != j and check_polygon_collision(vertices, other_vertices):
                            has_collisions = True
                            break
                    if has_collisions:
                        break
                
                if has_collisions:
                    complete_surface = font.render("Evolution Complete - No Valid Solution Found", True, (255, 0, 0))
                else:
                    complete_surface = font.render("Evolution Complete - Valid Solution Found!", True, (0, 100, 0))
                
                complete_rect = complete_surface.get_rect(center=(screen.get_width()/2, 40))
                screen.blit(complete_surface, complete_rect)
        
        # Draw current best solution if available
        if current_best_solution is not None:
            draw_solution(screen, boundary, polygons, current_best_solution, scale=4)
        
        # Update display
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()



if __name__ == "__main__":
    main()
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState
from numba import cuda, float32, int32
import math
from algorithms import polygon_helpers
from algorithms.polygon_helpers import compute_polygon_vertices_cpu, check_polygon_collision

class CUDAPolygonPacker:
    def __init__(self, 
                 boundary_polygon: np.ndarray,
                 regular_polygons: list[tuple[int, float]],
                 population_size: int = 1024,
                 mutation_rate: float = 0.1,
                 penalties: dict = None,
                 random_seed: int = None):
        """Initialize the CUDA-based polygon packing genetic algorithm."""
        self.boundary = boundary_polygon
        self.polygons = regular_polygons
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        
        # Initialize RNG with seed
        self.rng = RandomState(MT19937(seed=random_seed))
        if random_seed is not None:
            print(f"Initialized random number generator with seed: {random_seed}")
        
        # Set penalties from config or use defaults
        self.penalties = penalties or {
            'boundary_penalty': 10000.0,
            'overlap_penalty': 5000.0,
            'spacing_penalty': 1000.0
        }

        # Create penalty array for CUDA
        self.penalty_array = np.array([
            self.penalties['boundary_penalty'],
            self.penalties['overlap_penalty'],
            self.penalties['spacing_penalty']
        ], dtype=np.float32)

        print(f"Penalties: {self.penalty_array}")
        
        # Calculate boundary bounds for initialization and mutation
        self.min_x = np.min(boundary_polygon[:, 0])
        self.max_x = np.max(boundary_polygon[:, 0])
        self.min_y = np.min(boundary_polygon[:, 1])
        self.max_y = np.max(boundary_polygon[:, 1])
        
        self.num_polygons = len(regular_polygons)
        self.chromosome_length = self.num_polygons * 3
        
        # Initialize population with smarter positioning
        self.population = self._initialize_population()
        
        # Transfer data to GPU
        self.d_population = cuda.to_device(self.population)
        self.d_boundary = cuda.to_device(self.boundary)
        self.d_polygon_specs = cuda.to_device(np.array(regular_polygons))
        self.d_fitness = cuda.device_array(population_size, dtype=np.float32)
        self.d_penalties = cuda.to_device(self.penalty_array)
        
        self.threads_per_block = 256
        self.blocks = (population_size + self.threads_per_block - 1) // self.threads_per_block
    
    def _calculate_polygon_area(self, vertices):
        """Calculate area of a polygon using the shoelace formula."""
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return abs(area) / 2.0

    def get_utilization(self, solution):
        """Calculate actual space utilization for the best solution."""
        boundary_area = 0.0
        for i in range(len(self.boundary)):
            j = (i + 1) % len(self.boundary)
            boundary_area += (self.boundary[i][0] * self.boundary[j][1] - 
                            self.boundary[j][0] * self.boundary[i][1])
        boundary_area = abs(boundary_area) / 2.0
        
        # Calculate total area of polygons that are actually within the boundary
        valid_polygon_area = 0.0
        for i, (num_sides, size) in enumerate(self.polygons):
            # Get polygon position and rotation
            base_idx = i * 3
            x = solution[base_idx]
            y = solution[base_idx + 1]
            rotation = solution[base_idx + 2]
            
            # Compute vertices
            vertices = compute_polygon_vertices_cpu(x, y, rotation, num_sides, size)
            
            # Only count area if polygon is fully contained and not overlapping
            is_contained = check_polygon_collision(vertices, self.boundary, is_boundary=True)
            is_overlapping = False
            
            # Check for overlaps with other polygons
            for j, (other_sides, other_size) in enumerate(self.polygons):
                if i != j:
                    other_base_idx = j * 3
                    other_x = solution[other_base_idx]
                    other_y = solution[other_base_idx + 1]
                    other_rotation = solution[other_base_idx + 2]
                    other_vertices = compute_polygon_vertices_cpu(
                        other_x, other_y, other_rotation, other_sides, other_size)
                    
                    if check_polygon_collision(vertices, other_vertices):
                        is_overlapping = True
                        break
            
            # Only add area if polygon is valid
            if is_contained and not is_overlapping:
                # For regular polygon, area = (n * s * s * cot(π/n)) / 4
                # where n is number of sides and s is side length
                # For our case, size is radius (R), and side length s = 2R*sin(π/n)
                polygon_area = (num_sides * size * size * 
                            np.sin(2 * np.pi / num_sides)) / 2
                valid_polygon_area += polygon_area
        
        # Calculate utilization percentage
        utilization = (valid_polygon_area / boundary_area) * 100
        return min(100, max(0, utilization))  # Clamp between 0 and 100

    def _initialize_population(self) -> np.ndarray:
        """Initialize population with better spread of initial positions."""
        population = np.zeros((self.population_size, self.chromosome_length), dtype=np.float32)
        
        # Calculate usable area margins based on largest polygon
        max_size = max(polygon[1] for polygon in self.polygons)
        margin = max_size * 1.2  # Add some buffer
        
        # Adjust bounds to account for polygon sizes
        safe_min_x = self.min_x + margin
        safe_max_x = self.max_x - margin
        safe_min_y = self.min_y + margin
        safe_max_y = self.max_y - margin
        
        for i in range(self.population_size):
            for j in range(self.num_polygons):
                # Grid-based initial positioning
                grid_x = safe_min_x + (safe_max_x - safe_min_x) * (j % 3) / 2
                grid_y = safe_min_y + (safe_max_y - safe_min_y) * (j // 3) / 2
                
                # Add some random offset from grid position
                x = grid_x + self.rng.uniform(-margin/2, margin/2)
                y = grid_y + self.rng.uniform(-margin/2, margin/2)
                rotation = self.rng.uniform(0, 2 * np.pi)
                
                base_idx = j * 3
                population[i, base_idx] = x
                population[i, base_idx + 1] = y
                population[i, base_idx + 2] = rotation
                
        return population
    
    def evolve(self, generations: int):
        """Evolve the population with improved selection pressure."""
        best_fitness_history = []
        best_solution = None
        
        self.population = self.d_population.copy_to_host()
        
        for gen in range(generations):
            # Calculate fitness
            calculate_fitness_kernel[self.blocks, self.threads_per_block](
                self.d_population,
                self.d_polygon_specs,
                self.d_boundary,
                self.d_fitness,
                self.d_penalties  # Pass penalties to kernel
            )
            
            fitness = self.d_fitness.copy_to_host()
            
            # Track best solution
            current_best_idx = np.argmin(fitness)
            current_best_fitness = fitness[current_best_idx]
            best_fitness_history.append(current_best_fitness)
            
            if best_solution is None or current_best_fitness < best_fitness_history[-1]:
                best_solution = self.population[current_best_idx].copy()
            
            # Enhanced tournament selection
            parents = self._tournament_selection(fitness)
            d_parents = cuda.to_device(parents)
            
            d_offspring = cuda.device_array_like(self.d_population)
            
            # Generate random states
            rand_states = self.rng.random(self.population_size * 4)
            d_rand_states = cuda.to_device(rand_states)
            
            # Perform crossover and mutation
            crossover_kernel[self.blocks, self.threads_per_block](
                d_parents, d_offspring, d_rand_states,
                self.min_x, self.max_x, self.min_y, self.max_y)
            
            mutation_kernel[self.blocks, self.threads_per_block](
                d_offspring, self.mutation_rate, d_rand_states,
                self.min_x, self.max_x, self.min_y, self.max_y)
            
            # Elitism: Keep best solution
            self.d_population = d_offspring
            self.population = self.d_population.copy_to_host()
            self.population[0] = best_solution  # Preserve best solution
            
        return best_fitness_history, self.population

    def _tournament_selection(self, fitness: np.ndarray) -> np.ndarray:
        """Enhanced tournament selection with higher selection pressure."""
        parents = np.zeros((self.population_size * 2, self.chromosome_length), dtype=np.float32)
        
        tournament_size = 5
        for i in range(self.population_size * 2):
            candidates = self.rng.choice(
                self.population_size,
                size=tournament_size,
                replace=False
            )
            winner = candidates[np.argmin(fitness[candidates])]
            parents[i] = self.population[winner]
        
        return parents

@cuda.jit
def calculate_fitness_kernel(population, polygon_specs, boundary, fitness_results, penalties):
    """CUDA kernel for calculating fitness with penalties."""
    idx = cuda.grid(1)
    if idx >= population.shape[0]:
        return
        
    solution = population[idx]
    penalty = 0.0
    
    max_vertices = 12
    polygon1_vertices = cuda.local.array((max_vertices, 2), dtype=float32)
    polygon2_vertices = cuda.local.array((max_vertices, 2), dtype=float32)
    
    num_polygons = len(polygon_specs)
    
    # Use penalties from config array
    BOUNDARY_PENALTY = penalties[0]  # boundary_penalty
    OVERLAP_PENALTY = penalties[1]   # overlap_penalty
    SPACING_PENALTY = penalties[2]   # spacing_penalty
    
    for i in range(num_polygons):
        base_idx1 = i * 3
        num_sides1 = polygon_specs[i][0]
        
        if num_sides1 > max_vertices:
            penalty += 20000.0
            continue
            
        p1_x = solution[base_idx1]
        p1_y = solution[base_idx1 + 1]
        p1_rotation = solution[base_idx1 + 2]
        size1 = polygon_specs[i][1]
        
        polygon_helpers.compute_polygon_vertices(
            p1_x, p1_y, p1_rotation,
            num_sides1, size1,
            polygon1_vertices
        )
        
        # Boundary check
        if not polygon_helpers.is_contained(polygon1_vertices, num_sides1, boundary):
            penalty += BOUNDARY_PENALTY
        
        # Check overlaps and spacing
        for j in range(i + 1, num_polygons):
            base_idx2 = j * 3
            p2_x = solution[base_idx2]
            p2_y = solution[base_idx2 + 1]
            p2_rotation = solution[base_idx2 + 2]
            num_sides2 = polygon_specs[j][0]
            size2 = polygon_specs[j][1]
            
            # Calculate center distance
            dx = p2_x - p1_x
            dy = p2_y - p1_y
            dist = math.sqrt(dx*dx + dy*dy)
            min_spacing = (size1 + size2) * 1.1
            
            if dist < min_spacing:
                penalty += SPACING_PENALTY * (min_spacing - dist)
            
            polygon_helpers.compute_polygon_vertices(
                p2_x, p2_y, p2_rotation,
                min(num_sides2, max_vertices), size2,
                polygon2_vertices
            )
            
            if polygon_helpers.polygons_overlap(polygon1_vertices, num_sides1, 
                                             polygon2_vertices, num_sides2):
                penalty += OVERLAP_PENALTY
    
    fitness_results[idx] = penalty

@cuda.jit
def crossover_kernel(parents, offspring, rand_states, min_x, max_x, min_y, max_y):
    """Improved crossover with boundary enforcement."""
    idx = cuda.grid(1)
    if idx < offspring.shape[0]:
        parent1_idx = idx * 2
        parent2_idx = idx * 2 + 1
        
        # Arithmetic crossover with random weight
        weight = 0.5 + (rand_states[idx] - 0.5) * 0.5  # Weight between 0.25 and 0.75
        
        for i in range(0, offspring.shape[1], 3):
            # Position crossover with boundary enforcement
            x = parents[parent1_idx, i] * weight + parents[parent2_idx, i] * (1 - weight)
            y = parents[parent1_idx, i+1] * weight + parents[parent2_idx, i+1] * (1 - weight)
            
            # Enforce boundaries
            x = max(min_x, min(max_x, x))
            y = max(min_y, min(max_y, y))
            
            # Rotation crossover
            rotation = parents[parent1_idx, i+2] * weight + parents[parent2_idx, i+2] * (1 - weight)
            rotation = rotation % (2 * math.pi)  # Normalize rotation
            
            offspring[idx, i] = x
            offspring[idx, i+1] = y
            offspring[idx, i+2] = rotation

@cuda.jit
def mutation_kernel(population, mutation_rate, rand_states, min_x, max_x, min_y, max_y):
    """Enhanced mutation with adaptive rates and boundary enforcement."""
    idx = cuda.grid(1)
    if idx < population.shape[0]:
        for i in range(0, population.shape[1], 3):
            if rand_states[idx + i] < mutation_rate:
                # Position mutation
                dx = (rand_states[idx + population.shape[0]] - 0.5) * (max_x - min_x) * 0.1
                dy = (rand_states[idx + 2 * population.shape[0]] - 0.5) * (max_y - min_y) * 0.1
                
                # Update and enforce boundaries
                population[idx, i] = max(min_x, min(max_x, population[idx, i] + dx))
                population[idx, i+1] = max(min_y, min(max_y, population[idx, i+1] + dy))
                
                # Rotation mutation with variable magnitude
                dr = (rand_states[idx + 3 * population.shape[0]] - 0.5) * math.pi * 0.5
                population[idx, i+2] = (population[idx, i+2] + dr) % (2 * math.pi)
from numba import cuda, float32, int32
import math
import numpy as np
from typing import List, Tuple, Union


def compute_polygon_vertices_cpu(center_x: float, center_y: float, 
                               rotation: float, num_sides: int, size: float) -> np.ndarray:
    """
    Compute vertices of a regular polygon on CPU.

    Args:
        center_x (float): X coordinate of polygon center
        center_y (float): Y coordinate of polygon center
        rotation (float): Rotation angle in radians
        num_sides (int): Number of sides in the polygon
        size (float): Size (radius) of the polygon

    Returns:
        np.ndarray: Array of (x, y) coordinates for polygon vertices
    """
    vertices = []
    for i in range(num_sides):
        angle = rotation + 2.0 * np.pi * i / num_sides
        x = center_x + size * np.cos(angle)
        y = center_y + size * np.sin(angle)
        vertices.append((x, y))
    return np.array(vertices)

def check_polygon_collision(vertices1: np.ndarray, 
                          vertices2: Union[np.ndarray, List[Tuple[float, float]]], 
                          is_boundary: bool = False) -> bool:
    """
    Check for collision between two polygons using Separating Axis Theorem.

    Args:
        vertices1 (np.ndarray): Vertices of first polygon
        vertices2 (Union[np.ndarray, List[Tuple[float, float]]]): Vertices of second polygon
        is_boundary (bool, optional): If True, checks if vertices1 is inside vertices2. Defaults to False

    Returns:
        bool: True if polygons collide (or vertices1 is outside vertices2 if is_boundary=True)
    """
    vertices2 = np.array(vertices2)
    
    def get_axes(vertices: np.ndarray) -> List[np.ndarray]:
        """Get the axes for SAT collision detection."""
        axes = []
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / np.linalg.norm(normal)
            axes.append(normal)
        return axes
    
    def project(vertices: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
        """Project vertices onto an axis."""
        dots = vertices.dot(axis)
        return min(dots), max(dots)
    
    # Get axes for both polygons
    axes = get_axes(vertices1)
    if not is_boundary:
        axes.extend(get_axes(vertices2))
    
    # Check projection overlap on each axis
    for axis in axes:
        min1, max1 = project(vertices1, axis)
        min2, max2 = project(vertices2, axis)
        
        if is_boundary:
            # For boundary check, polygon1 must be completely inside polygon2
            if min1 < min2 or max1 > max2:
                return False
        else:
            # For collision check, no overlap means no collision
            if max1 < min2 or max2 < min1:
                return False
    
    return True

@cuda.jit(device=True)
def compute_polygon_vertices(center_x: float, center_y: float, 
                           rotation: float, num_sides: int, size: float,
                           vertices_out: np.ndarray):
    """
    Compute vertices of a regular polygon.
    
    Args:
        center_x, center_y: Center position of the polygon
        rotation: Rotation angle in radians
        num_sides: Number of sides in the polygon
        size: Size (radius) of the polygon
        vertices_out: Output array to store vertices (Nx2)
    """
    for i in range(num_sides):
        angle = rotation + 2.0 * math.pi * i / num_sides
        vertices_out[i, 0] = center_x + size * math.cos(angle)
        vertices_out[i, 1] = center_y + size * math.sin(angle)

@cuda.jit(device=True)
def is_contained(polygon_vertices: np.ndarray, num_vertices: int, 
                boundary_vertices: np.ndarray) -> bool:
    """
    Check if a polygon is completely contained within the boundary polygon.
    Uses ray casting algorithm for point-in-polygon test.
    """
    # Check each vertex of the polygon
    for i in range(num_vertices):
        point_x = polygon_vertices[i, 0]
        point_y = polygon_vertices[i, 1]
        
        inside = False
        j = boundary_vertices.shape[0] - 1
        
        # Ray casting algorithm
        for k in range(boundary_vertices.shape[0]):
            if (((boundary_vertices[k, 1] > point_y) != 
                 (boundary_vertices[j, 1] > point_y)) and
                (point_x < (boundary_vertices[j, 0] - boundary_vertices[k, 0]) * 
                 (point_y - boundary_vertices[k, 1]) / 
                 (boundary_vertices[j, 1] - boundary_vertices[k, 1]) + 
                 boundary_vertices[k, 0])):
                inside = not inside
            j = k
            
        if not inside:
            return False
    
    return True

@cuda.jit(device=True)
def polygons_overlap(vertices1: np.ndarray, num_vertices1: int,
                    vertices2: np.ndarray, num_vertices2: int) -> bool:
    """
    Check if two polygons overlap using the Separating Axis Theorem (SAT).
    """
    # Check axes from first polygon
    for i in range(num_vertices1):
        j = (i + 1) % num_vertices1
        
        # Convert indices to integers explicitly
        i_int = int32(i)  # Add explicit conversion
        j_int = int32(j)  # Add explicit conversion
        
        # Calculate edge vector using integer indices
        edge_x = vertices1[j_int, 0] - vertices1[i_int, 0]
        edge_y = vertices1[j_int, 1] - vertices1[i_int, 1]
        
        # Calculate normal (perpendicular) vector
        normal_x = -edge_y
        normal_y = edge_x
        
        # Normalize the normal vector
        length = math.sqrt(normal_x * normal_x + normal_y * normal_y)
        if length < 1e-6:  # Avoid division by zero
            continue
        normal_x /= length
        normal_y /= length
        
        # Project both polygons onto the normal
        min1, max1 = project_polygon(vertices1, num_vertices1, normal_x, normal_y)
        min2, max2 = project_polygon(vertices2, num_vertices2, normal_x, normal_y)
        
        # Check for separation
        if max1 < min2 or max2 < min1:
            return False
    
    # Check axes from second polygon
    for i in range(num_vertices2):
        j = (i + 1) % num_vertices2
        
        # Convert indices to integers explicitly
        i_int = int32(i)  # Add explicit conversion
        j_int = int32(j)  # Add explicit conversion
        
        edge_x = vertices2[j_int, 0] - vertices2[i_int, 0]
        edge_y = vertices2[j_int, 1] - vertices2[i_int, 1]
        
        normal_x = -edge_y
        normal_y = edge_x
        
        length = math.sqrt(normal_x * normal_x + normal_y * normal_y)
        if length < 1e-6:
            continue
        normal_x /= length
        normal_y /= length
        
        min1, max1 = project_polygon(vertices1, num_vertices1, normal_x, normal_y)
        min2, max2 = project_polygon(vertices2, num_vertices2, normal_x, normal_y)
        
        if max1 < min2 or max2 < min1:
            return False
    
    return True

@cuda.jit(device=True)
def project_polygon(vertices: np.ndarray, num_vertices: int, 
                   normal_x: float, normal_y: float) -> tuple[float, float]:
    """Project a polygon onto a normal vector and return min/max values."""
    min_proj = float('inf')
    max_proj = float('-inf')
    
    for i in range(num_vertices):
        i_int = int32(i)  # Add explicit conversion
        projection = vertices[i_int, 0] * normal_x + vertices[i_int, 1] * normal_y
        min_proj = min(min_proj, projection)
        max_proj = max(max_proj, projection)
    
    return min_proj, max_proj


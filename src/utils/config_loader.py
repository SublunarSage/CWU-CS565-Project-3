import yaml
import numpy as np

def load_config(config_path: str):
    """    
    Load configuration from YAML file.

    Args:
        config_path (str): Path to the YAML configuration file

    Returns:
        tuple containing:
            - np.ndarray: Boundary polygon vertices
            - list: List of (num_sides, size) tuples defining polygons
            - dict: Genetic algorithm parameters
            - dict: Penalty weights
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert boundary to numpy array
    boundary = np.array(config['boundary'], dtype=np.float32)
    
    # Convert polygons to list of tuples
    polygons = [(int(sides), float(size)) for sides, size in config['polygons']]
    
    # Extract GA parameters
    ga_params = config['ga_params']
    
    # Ensure random_seed exists in ga_params (default to None if not specified)
    if 'random_seed' not in ga_params:
        ga_params['random_seed'] = None

    # Extract penalties
    penalties = config['penalties']
    
    return boundary, polygons, ga_params, penalties
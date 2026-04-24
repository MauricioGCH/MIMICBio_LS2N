"""
Configuraciones para grid search.
Lee desde config.yaml y genera combinaciones automáticamente.
"""
import yaml
import itertools
from copy import deepcopy


def load_base_config(config_path='config.yaml'):
    """Carga configuración base desde YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def extract_grid_params(config, exclude_keys=None):
    """
    Extrae parámetros que son listas (grid search).
    """
    if exclude_keys is None:
        exclude_keys = ['t_R', 't0', 'beta', 'channel', 'data_path', 'output_dir']
    
    grid_params = {}
    fixed_params = {}
    
    for key, value in config.items():
        if key in exclude_keys:
            fixed_params[key] = value
        elif isinstance(value, list) and len(value) > 1:
            grid_params[key] = value  # ← Solo si es lista con múltiples elementos
        else:
            fixed_params[key] = value
    
    return grid_params, fixed_params


def generate_grid_combinations(grid_params):
    """Genera todas las combinaciones del grid."""
    if not grid_params:
        return [{}]
    
    keys = list(grid_params.keys())
    values = list(grid_params.values())
    
    combinaciones = []
    for combo in itertools.product(*values):
        combinaciones.append(dict(zip(keys, combo)))
    
    return combinaciones


def get_all_configs(config_path='config.yaml'):
    """
    Genera lista completa de configuraciones.
    """
    base_config = load_base_config(config_path)
    grid_params, fixed_params = extract_grid_params(base_config)
    
    if not grid_params:
        return [fixed_params]
    
    grid_combos = generate_grid_combinations(grid_params)
    
    configs = []
    for combo in grid_combos:
        config = deepcopy(fixed_params)
        config.update(combo)
        configs.append(config)
    
    return configs
"""
Actualiza el Excel de resumen de experimentos después de cada corrida.
Versión CORREGIDA que NO sobrescribe filas anteriores.
"""
import os
import pandas as pd
from pathlib import Path


def load_existing_summary(excel_path):
    """Carga el Excel existente si existe, sino retorna DataFrame vacío."""
    if os.path.exists(excel_path) and os.path.getsize(excel_path) > 0:
        try:
            df = pd.read_excel(excel_path, engine='openpyxl')
            if len(df) > 0:
                return df
        except Exception as e:
            print(f"⚠️ Error leyendo Excel: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def save_summary(df, excel_path):
    """Guarda el DataFrame en Excel."""
    df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"📊 Resumen de experimentos actualizado: {excel_path} (total: {len(df)} experimentos)")


def get_experiment_info(exp_path, config):
    """
    Extrae la información relevante de un experimento.
    """
    is_sintetic = config.get('Sintetic', False)
    
    if is_sintetic:
        data_source = "SINTETIC"
    else:
        data_source = config.get('data_path', 'N/A')
    
    channels = config.get('channel', 'N/A')
    if isinstance(channels, list):
        channels = str(channels)
    
    threshold_method = config.get('threshold_method', 'MAD')
    
    info = {
        'exp_folder': os.path.basename(exp_path),
        'exp_path': exp_path,
        'Sintetic': is_sintetic,
        'data_path': data_source,
        'channel': channels,
        'sampling_rate': config.get('sampling_rate', 'N/A'),
        'skip_s': config.get('skip_s', 'N/A'),
        'length_s': config.get('length_s', 'N/A'),
        'sampling_rate_DS': config.get('sampling_rate_DS', 'N/A'),
        'init_data': config.get('init_data', 'N/A'),
        'n_s': config.get('n_s', 'N/A'),
        'weibull_init_method': config.get('weibull_init_method', 'N/A'),
        'H_update': config.get('H_update', 'N/A'),
        'threshold_method': threshold_method,
    }
    
    return info


def update_experiment_summary(exp_path, config, base_output_dir, excel_name="experiments_summary.xlsx"):
    """
    Actualiza el Excel de resumen con la información del experimento actual.
    NO sobrescribe experimentos anteriores.
    
    Parámetros:
    - exp_path: ruta de la carpeta del experimento
    - config: configuración del experimento
    - base_output_dir: directorio base donde se guardan los experimentos
    - excel_name: nombre del archivo Excel
    """
    
    excel_path = os.path.join(base_output_dir, excel_name)
    
    # Cargar Excel existente
    df_existing = load_existing_summary(excel_path)
    
    # Obtener información del experimento actual
    new_entry = get_experiment_info(exp_path, config)
    
    # 🔥 IMPORTANTE: Verificar si este experimento YA EXISTE por carpeta
    exp_folder_name = new_entry['exp_folder']
    
    if not df_existing.empty and 'Carpeta Experimento' in df_existing.columns:
        # Verificar si ya existe este experimento (usando el nombre correcto de columna)
        if exp_folder_name in df_existing['Carpeta Experimento'].values:
            print(f"⚠️ Experimento {exp_folder_name} ya existe en el resumen. Saltando...")
            return df_existing
    
    # Crear nuevo registro como DataFrame
    df_new = pd.DataFrame([new_entry])
    
    # Renombrar columnas para el Excel
    df_new = df_new.rename(columns={
        'exp_folder': 'Carpeta Experimento',
        'exp_path': 'Ruta Completa',
        'Sintetic': '¿Sintético?',
        'data_path': 'Data Path / Fuente',
        'channel': 'Canal(es)',
        'sampling_rate': 'Fs (Hz)',
        'skip_s': 'Skip (s)',
        'length_s': 'Length (s)',
        'sampling_rate_DS': 'Downsampling (Hz)',
        'init_data': 'Init Data (muestras)',
        'n_s': 'n_s (beam width)',
        'weibull_init_method': 'Weibull Init Method',
        'H_update': 'H Update Method',
        'threshold_method': 'Threshold Method',
    })
    
    # 🔥 CONCATENAR (no sobrescribir)
    if df_existing.empty:
        df_final = df_new
        print(f"➕ Creando nuevo resumen con experimento: {exp_folder_name}")
    else:
        # Asegurar que las columnas coincidan
        for col in df_new.columns:
            if col not in df_existing.columns:
                df_existing[col] = 'N/A'
        for col in df_existing.columns:
            if col not in df_new.columns:
                df_new[col] = 'N/A'
        
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
        print(f"➕ Añadido experimento: {exp_folder_name}")
    
    # Ordenar por nombre de carpeta (fecha)
    df_final = df_final.sort_values('Carpeta Experimento').reset_index(drop=True)
    
    # Guardar
    save_summary(df_final, excel_path)
    
    return df_final
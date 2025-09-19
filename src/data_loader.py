"""
Módulo para carga y preparación de datos del portafolio de inversión.
Funcionalidad 0: Carga y Preparación de Datos
"""

import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import io


class DataLoader:
    """
    Clase para manejar la carga y validación de datos del portafolio.
    """
    
    def __init__(self, data_path: str = "datos/portafolio_21_activos.csv"):
        """
        Inicializa el cargador de datos.
        
        Args:
            data_path (str): Ruta al archivo CSV con los datos
        """
        self.data_path = Path(data_path)
        self.data = None
        self.valid_assets = None
        self.excluded_assets = []
        self.data_source = "file"  # "file" o "uploaded"
        
    def load_data(self) -> bool:
        """
        Carga los datos del archivo CSV y realiza validaciones básicas.
        
        Returns:
            bool: True si la carga fue exitosa, False en caso contrario
        """
        try:
            # Cargar datos
            self.data = pd.read_csv(self.data_path)
            self.data_source = "file"
            
            # Validar estructura básica
            if not self._validate_structure():
                return False
                
            # Procesar fechas
            self._process_dates()
            
            # Validar datos faltantes
            self._validate_missing_data()
            
            return True
            
        except FileNotFoundError:
            st.error(f"❌ Error: No se encontró el archivo {self.data_path}")
            return False
        except Exception as e:
            st.error(f"❌ Error al cargar los datos: {str(e)}")
            return False
    
    def load_data_from_uploaded_file(self, uploaded_file) -> bool:
        """
        Carga los datos desde un archivo CSV cargado por el usuario.
        
        Args:
            uploaded_file: Archivo CSV cargado mediante st.file_uploader
            
        Returns:
            bool: True si la carga fue exitosa, False en caso contrario
        """
        try:
            # Leer el archivo cargado
            self.data = pd.read_csv(uploaded_file)
            self.data_source = "uploaded"
            
            # Validar estructura básica
            if not self._validate_structure():
                return False
                
            # Procesar fechas
            self._process_dates()
            
            # Validar datos faltantes
            self._validate_missing_data()
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo cargado: {str(e)}")
            return False
    
    def _validate_structure(self) -> bool:
        """
        Valida la estructura básica del archivo de datos.
        
        Returns:
            bool: True si la estructura es válida
        """
        if self.data is None:
            return False
            
        # Verificar que existe la columna Date
        if 'Date' not in self.data.columns:
            st.error("❌ Error: No se encontró la columna 'Date' en los datos")
            return False
            
        # Verificar que hay al menos 2 columnas (Date + al menos 1 activo)
        if len(self.data.columns) < 2:
            st.error("❌ Error: El archivo debe contener al menos una columna de fecha y una de activos")
            return False
            
        # Validaciones adicionales para estructura esperada
        asset_columns = [col for col in self.data.columns if col != 'Date']
        
        # Verificar que hay datos suficientes (al menos 50 observaciones para análisis robusto)
        if len(self.data) < 50:
            st.warning(f"⚠️ Advertencia: El archivo contiene solo {len(self.data)} observaciones. Se recomiendan al menos 50 para un análisis robusto.")
        
        # Verificar que hay suficientes activos (al menos 5 para diversificación)
        if len(asset_columns) < 5:
            st.warning(f"⚠️ Advertencia: El archivo contiene solo {len(asset_columns)} activos. Se recomiendan al menos 5 para diversificación.")
            
        return True
    
    def _process_dates(self):
        """
        Procesa y valida las fechas del dataset.
        """
        try:
            # Convertir a datetime
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            
            # Ordenar por fecha
            self.data = self.data.sort_values('Date').reset_index(drop=True)
            
        except Exception as e:
            st.error(f"❌ Error al procesar las fechas: {str(e)}")
            raise
    
    def _validate_missing_data(self):
        """
        Valida datos faltantes y excluye activos con datos incompletos.
        """
        # Obtener columnas de activos (excluyendo Date)
        asset_columns = [col for col in self.data.columns if col != 'Date']
        
        # Verificar datos faltantes por activo
        missing_data = self.data[asset_columns].isnull().sum()
        
        # Identificar activos con datos faltantes
        assets_with_missing = missing_data[missing_data > 0]
        
        if len(assets_with_missing) > 0:
            # Excluir activos con datos faltantes
            self.excluded_assets = assets_with_missing.index.tolist()
            self.valid_assets = [col for col in asset_columns if col not in self.excluded_assets]
            
            # Mostrar mensajes informativos
            for asset in self.excluded_assets:
                missing_count = missing_data[asset]
                st.warning(f"⚠️ Se omitió la empresa {asset} por {missing_count} datos faltantes")
        else:
            # Todos los activos son válidos
            self.valid_assets = asset_columns
            self.excluded_assets = []
    
    def get_data_summary(self) -> dict:
        """
        Obtiene un resumen de los datos cargados.
        
        Returns:
            dict: Diccionario con información del dataset
        """
        if self.data is None:
            return {}
            
        return {
            'total_assets': len(self.valid_assets),
            'excluded_assets': len(self.excluded_assets),
            'date_range': {
                'start': self.data['Date'].min().strftime('%Y-%m-%d'),
                'end': self.data['Date'].max().strftime('%Y-%m-%d')
            },
            'observations': len(self.data),
            'valid_assets': self.valid_assets,
            'excluded_assets_list': self.excluded_assets,
            'data_source': self.data_source
        }
    
    def get_asset_data(self, asset: str) -> pd.Series:
        """
        Obtiene los datos de un activo específico.
        
        Args:
            asset (str): Ticker del activo
            
        Returns:
            pd.Series: Serie temporal de precios del activo
        """
        if self.data is None or asset not in self.data.columns:
            raise ValueError(f"Activo {asset} no encontrado en los datos")
            
        return self.data[asset]
    
    def get_all_asset_data(self) -> pd.DataFrame:
        """
        Obtiene los datos de todos los activos válidos.
        
        Returns:
            pd.DataFrame: DataFrame con fechas y precios de activos válidos
        """
        if self.data is None:
            return pd.DataFrame()
            
        # Retornar solo activos válidos
        columns_to_return = ['Date'] + self.valid_assets
        return self.data[columns_to_return].copy()
    
    def get_dates(self) -> pd.Series:
        """
        Obtiene las fechas del dataset.
        
        Returns:
            pd.Series: Serie de fechas
        """
        if self.data is None:
            return pd.Series()
            
        return self.data['Date'].copy()


def load_portfolio_data() -> DataLoader:
    """
    Función de conveniencia para cargar los datos del portafolio.
    
    Returns:
        DataLoader: Instancia del cargador con datos cargados
    """
    loader = DataLoader()
    
    if loader.load_data():
        return loader
    else:
        raise RuntimeError("No se pudieron cargar los datos del portafolio")

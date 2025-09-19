"""
Módulo para cálculos cuantitativos del portafolio de inversión.
FASE 2 DEL PROYECTO: Backend de Cálculos Cuantitativos
IMPORTANTE: Todavía NO se hace optimización - solo fórmulas auxiliares
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import streamlit as st
from scipy.optimize import minimize


class QuantitativeCalculations:
    """
    Clase para manejar los cálculos cuantitativos del portafolio.
    FASE 2: Backend de Cálculos Cuantitativos (sin optimización)
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Inicializa los cálculos cuantitativos.
        
        Args:
            data (pd.DataFrame): DataFrame con precios históricos (fecha + activos)
        """
        self.data = data.copy()
        self.prices = data.drop('Date', axis=1)  # Solo precios, sin fechas
        self.returns = None
        self.expected_returns = None
        self.covariance_matrix = None
        self.correlation_matrix = None
        
    def calcular_rendimientos(self) -> pd.DataFrame:
        """
        Calcula los rendimientos históricos mensuales a partir de los precios.
        FASE 2: Fórmula auxiliar para rendimientos
        
        Returns:
            pd.DataFrame: DataFrame con rendimientos mensuales
        """
        # Calcular rendimientos mensuales usando la fórmula: (P_t / P_{t-1}) - 1
        self.returns = self.prices.pct_change().dropna()
        
        return self.returns
    
    def calcular_rendimiento_esperado_anualizado(self, method: str = 'multiplicative') -> pd.Series:
        """
        Calcula el rendimiento esperado anualizado usando fórmula multiplicativa.
        FASE 2: Fórmula auxiliar para rendimiento esperado
        
        Args:
            method (str): Método de cálculo ('multiplicative' o 'simple')
            
        Returns:
            pd.Series: Rendimientos esperados anualizados por activo
        """
        if self.returns is None:
            self.calcular_rendimientos()
        
        if method == 'multiplicative':
            # Fórmula multiplicativa: (1 + r_mensual)^12 - 1
            monthly_mean = self.returns.mean()
            self.expected_returns = (1 + monthly_mean) ** 12 - 1
        else:
            # Fórmula simple: r_mensual * 12 (menos precisa)
            self.expected_returns = self.returns.mean() * 12
        
        return self.expected_returns
    
    def calcular_volatilidad_anualizada(self, method: str = 'multiplicative') -> pd.Series:
        """
        Calcula la volatilidad anualizada usando fórmula multiplicativa.
        FASE 2: Fórmula auxiliar para volatilidad
        
        Args:
            method (str): Método de cálculo ('multiplicative' o 'simple')
            
        Returns:
            pd.Series: Volatilidades anualizadas por activo
        """
        if self.returns is None:
            self.calcular_rendimientos()
        
        if method == 'multiplicative':
            # Fórmula multiplicativa: σ_mensual * √12
            monthly_std = self.returns.std()
            volatility = monthly_std * np.sqrt(12)
        else:
            # Fórmula simple: σ_mensual * 12 (incorrecta)
            volatility = self.returns.std() * 12
        
        return volatility
    
    def calcular_matriz_covarianzas(self) -> pd.DataFrame:
        """
        Calcula la matriz de covarianzas de los activos.
        FASE 2: Fórmula auxiliar para matriz de covarianzas
        
        Returns:
            pd.DataFrame: Matriz de covarianzas
        """
        if self.returns is None:
            self.calcular_rendimientos()
        
        # Calcular matriz de covarianzas mensual
        monthly_cov = self.returns.cov()
        
        # Anualizar la matriz de covarianzas: Cov_anual = Cov_mensual * 12
        self.covariance_matrix = monthly_cov * 12
        
        return self.covariance_matrix
    
    def calcular_matriz_correlaciones(self) -> pd.DataFrame:
        """
        Calcula la matriz de correlaciones de los activos.
        FASE 2: Fórmula auxiliar para matriz de correlaciones
        
        Returns:
            pd.DataFrame: Matriz de correlaciones
        """
        if self.returns is None:
            self.calcular_rendimientos()
        
        # La matriz de correlaciones no cambia al anualizar
        self.correlation_matrix = self.returns.corr()
        
        return self.correlation_matrix
    
    def calcular_metricas_por_activo(self) -> pd.DataFrame:
        """
        Calcula todas las métricas principales por activo.
        FASE 2: Resumen de fórmulas auxiliares
        
        Returns:
            pd.DataFrame: DataFrame con métricas por activo
        """
        # Calcular todas las métricas
        returns_annual = self.calcular_rendimiento_esperado_anualizado()
        volatility_annual = self.calcular_volatilidad_anualizada()
        
        # Crear DataFrame con métricas
        metrics_df = pd.DataFrame({
            'Rendimiento_Esperado_Anual': returns_annual,
            'Volatilidad_Anual': volatility_annual,
            'Ratio_Riesgo_Rendimiento': returns_annual / volatility_annual
        })
        
        # Agregar métricas adicionales
        if self.returns is None:
            self.calcular_rendimientos()
        
        metrics_df['Rendimiento_Mensual_Promedio'] = self.returns.mean()
        metrics_df['Volatilidad_Mensual'] = self.returns.std()
        metrics_df['Observaciones'] = self.returns.count()
        
        return metrics_df
    
    def calcular_metricas_portafolio_pesos_iguales(self) -> Dict:
        """
        Calcula métricas para un portafolio de pesos iguales.
        FASE 2: Fórmula auxiliar para portafolio de referencia
        
        Returns:
            Dict: Diccionario con métricas del portafolio
        """
        if self.returns is None:
            self.calcular_rendimientos()
        
        # Pesos iguales
        n_assets = len(self.returns.columns)
        equal_weights = np.array([1/n_assets] * n_assets)
        
        # Rendimiento del portafolio
        portfolio_returns = self.returns.dot(equal_weights)
        
        # Métricas del portafolio
        portfolio_metrics = {
            'rendimiento_esperado_anual': self.calcular_rendimiento_esperado_anualizado().dot(equal_weights),
            'volatilidad_anual': np.sqrt(equal_weights.T @ self.calcular_matriz_covarianzas() @ equal_weights),
            'rendimiento_mensual_promedio': portfolio_returns.mean(),
            'volatilidad_mensual': portfolio_returns.std(),
            'observaciones': len(portfolio_returns)
        }
        
        return portfolio_metrics
    
    def optimizar_portafolio_sharpe(self, tasa_libre_riesgo: float = 0.04, activos_seleccionados: List[str] = None) -> Dict:
        """
        Optimiza el portafolio para maximizar el ratio de Sharpe usando Python puro.
        Implementación de optimización de Markowitz sin librerías externas.
        
        Args:
            tasa_libre_riesgo (float): Tasa libre de riesgo anual (default: 4%)
            activos_seleccionados (List[str]): Lista de activos a optimizar (opcional)
            
        Returns:
            Dict: Diccionario con pesos óptimos y métricas del portafolio optimizado
        """
        if self.returns is None:
            self.calcular_rendimientos()
        
        # Si no se especifican activos, usar todos
        if activos_seleccionados is None:
            activos_seleccionados = self.returns.columns.tolist()
        
        # Filtrar datos solo para activos seleccionados
        returns_selected = self.returns[activos_seleccionados]
        expected_returns_selected = self.calcular_rendimiento_esperado_anualizado()[activos_seleccionados]
        cov_matrix_selected = self.calcular_matriz_covarianzas().loc[activos_seleccionados, activos_seleccionados]
        
        n_assets = len(activos_seleccionados)
        
        # Función objetivo: minimizar el negativo del ratio de Sharpe
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns_selected)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_selected, weights)))
            sharpe_ratio = (portfolio_return - tasa_libre_riesgo) / portfolio_vol
            return -sharpe_ratio  # Negativo porque minimizamos
        
        # Restricciones
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Suma de pesos = 1
        bounds = tuple((0, 1) for _ in range(n_assets))  # Pesos entre 0 y 1
        
        # Punto inicial: pesos iguales
        initial_guess = np.array([1/n_assets] * n_assets)
        
        # Optimización
        result = minimize(objective, initial_guess, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            
            # Calcular métricas del portafolio optimizado
            portfolio_return = np.dot(optimal_weights, expected_returns_selected)
            portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix_selected, optimal_weights)))
            sharpe_ratio = (portfolio_return - tasa_libre_riesgo) / portfolio_vol
            
            return {
                'pesos': optimal_weights,
                'rendimiento_esperado_anual': portfolio_return,
                'volatilidad_anual': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'tasa_libre_riesgo': tasa_libre_riesgo,
                'optimizacion_exitosa': True,
                'activos': activos_seleccionados
            }
        else:
            return {
                'optimizacion_exitosa': False,
                'error': result.message
            }
    
    def optimizar_portafolio_con_restricciones(self, tasa_libre_riesgo: float = 0.04, 
                                             activos_seleccionados: List[str] = None, 
                                             pesos_forzados: Dict = None) -> Dict:
        """
        Optimiza el portafolio con restricciones de pesos forzados.
        
        Args:
            tasa_libre_riesgo (float): Tasa libre de riesgo anual
            activos_seleccionados (List[str]): Lista de activos a optimizar
            pesos_forzados (Dict): Diccionario con pesos forzados {activo: peso}
            
        Returns:
            Dict: Diccionario con pesos óptimos y métricas del portafolio con restricciones
        """
        if self.returns is None:
            self.calcular_rendimientos()
        
        if activos_seleccionados is None:
            activos_seleccionados = self.returns.columns.tolist()
        
        if pesos_forzados is None:
            pesos_forzados = {}
        
        # Filtrar datos solo para activos seleccionados
        returns_selected = self.returns[activos_seleccionados]
        expected_returns_selected = self.calcular_rendimiento_esperado_anualizado()[activos_seleccionados]
        cov_matrix_selected = self.calcular_matriz_covarianzas().loc[activos_seleccionados, activos_seleccionados]
        
        n_assets = len(activos_seleccionados)
        
        # Función objetivo: minimizar el negativo del ratio de Sharpe
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns_selected)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_selected, weights)))
            sharpe_ratio = (portfolio_return - tasa_libre_riesgo) / portfolio_vol
            return -sharpe_ratio
        
        # Restricciones base: suma de pesos = 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Agregar restricciones de pesos forzados
        for i, asset in enumerate(activos_seleccionados):
            if asset in pesos_forzados:
                peso_forzado = pesos_forzados[asset]
                constraints.append({
                    'type': 'eq', 
                    'fun': lambda x, idx=i, peso=peso_forzado: x[idx] - peso
                })
        
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Punto inicial considerando pesos forzados
        initial_guess = np.array([1/n_assets] * n_assets)
        for i, asset in enumerate(activos_seleccionados):
            if asset in pesos_forzados:
                initial_guess[i] = pesos_forzados[asset]
        
        # Renormalizar punto inicial
        if np.sum(initial_guess) > 0:
            initial_guess = initial_guess / np.sum(initial_guess)
        
        # Optimización
        result = minimize(objective, initial_guess, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            
            # Calcular métricas del portafolio optimizado con restricciones
            portfolio_return = np.dot(optimal_weights, expected_returns_selected)
            portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix_selected, optimal_weights)))
            sharpe_ratio = (portfolio_return - tasa_libre_riesgo) / portfolio_vol
            
            return {
                'pesos': optimal_weights,
                'rendimiento_esperado_anual': portfolio_return,
                'volatilidad_anual': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'tasa_libre_riesgo': tasa_libre_riesgo,
                'optimizacion_exitosa': True,
                'activos': activos_seleccionados,
                'pesos_forzados': pesos_forzados
            }
        else:
            return {
                'optimizacion_exitosa': False,
                'error': result.message
            }
    
    def calcular_sharpe_ratio(self, rendimiento_anual: float, volatilidad_anual: float, 
                            tasa_libre_riesgo: float = 0.04) -> float:
        """
        Calcula el ratio de Sharpe para un portafolio.
        
        Args:
            rendimiento_anual (float): Rendimiento esperado anual
            volatilidad_anual (float): Volatilidad anual
            tasa_libre_riesgo (float): Tasa libre de riesgo anual
            
        Returns:
            float: Ratio de Sharpe
        """
        return (rendimiento_anual - tasa_libre_riesgo) / volatilidad_anual
    
    def calcular_frontera_eficiente(self, n_puntos: int = 50, activos_seleccionados: List[str] = None) -> Dict:
        """
        Calcula la frontera eficiente de Markowitz usando optimización cuadrática.
        
        Args:
            n_puntos (int): Número de puntos para la frontera eficiente
            activos_seleccionados (List[str]): Lista de activos a considerar en la frontera
            
        Returns:
            Dict: Diccionario con rendimientos, volatilidades y pesos de la frontera
        """
        if self.returns is None:
            self.calcular_rendimientos()
        
        # Obtener rendimientos esperados y matriz de covarianzas
        expected_returns_full = self.calcular_rendimiento_esperado_anualizado()
        cov_matrix_full = self.calcular_matriz_covarianzas()
        
        # Filtrar por activos seleccionados si se especifican
        if activos_seleccionados is not None:
            expected_returns = expected_returns_full[activos_seleccionados]
            cov_matrix = cov_matrix_full.loc[activos_seleccionados, activos_seleccionados]
        else:
            expected_returns = expected_returns_full
            cov_matrix = cov_matrix_full
        
        n_assets = len(expected_returns)
        
        # Definir rango de rendimientos objetivo
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_puntos)
        
        # Listas para almacenar resultados
        efficient_returns = []
        efficient_volatilities = []
        efficient_weights = []
        
        # Función objetivo: minimizar la varianza del portafolio
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Restricciones base: suma de pesos = 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        for target_return in target_returns:
            # Agregar restricción de rendimiento objetivo
            return_constraint = {
                'type': 'eq', 
                'fun': lambda x, target=target_return: np.dot(x, expected_returns) - target
            }
            current_constraints = constraints + [return_constraint]
            
            # Punto inicial: pesos iguales
            initial_guess = np.array([1/n_assets] * n_assets)
            
            # Optimización
            result = minimize(objective, initial_guess, method='SLSQP',
                            bounds=bounds, constraints=current_constraints)
            
            if result.success:
                weights = result.x
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                efficient_returns.append(portfolio_return)
                efficient_volatilities.append(portfolio_vol)
                efficient_weights.append(weights)
        
        return {
            'rendimientos': np.array(efficient_returns),
            'volatilidades': np.array(efficient_volatilities),
            'pesos': np.array(efficient_weights),
            'exito': len(efficient_returns) > 0
        }
    
    def calcular_cml(self, rendimiento_mercado: float, volatilidad_mercado: float, 
                    tasa_libre_riesgo: float = 0.04, n_puntos: int = 100) -> Dict:
        """
        Calcula la línea del mercado de capitales (CML).
        
        Args:
            rendimiento_mercado (float): Rendimiento del portafolio de mercado
            volatilidad_mercado (float): Volatilidad del portafolio de mercado
            tasa_libre_riesgo (float): Tasa libre de riesgo anual
            n_puntos (int): Número de puntos para la línea CML
            
        Returns:
            Dict: Diccionario con puntos de la CML
        """
        # Calcular la pendiente de la CML (ratio de Sharpe del mercado)
        sharpe_mercado = (rendimiento_mercado - tasa_libre_riesgo) / volatilidad_mercado
        
        # Rango de volatilidades para la CML
        max_vol = volatilidad_mercado * 2  # Extender más allá del portafolio de mercado
        volatilidades_cml = np.linspace(0, max_vol, n_puntos)
        
        # Calcular rendimientos de la CML: R = Rf + (Rm - Rf) * σ / σm
        rendimientos_cml = tasa_libre_riesgo + sharpe_mercado * volatilidades_cml
        
        return {
            'volatilidades': volatilidades_cml,
            'rendimientos': rendimientos_cml,
            'pendiente': sharpe_mercado,
            'punto_tangente': {
                'volatilidad': volatilidad_mercado,
                'rendimiento': rendimiento_mercado
            }
        }
    
    def calcular_evolucion_historica(self, pesos_portafolio: np.ndarray, activos_seleccionados: List[str], 
                                   valor_inicial: float = 100.0) -> pd.Series:
        """
        Calcula la evolución histórica de un portafolio con pesos dados.
        
        Args:
            pesos_portafolio (np.ndarray): Pesos del portafolio
            activos_seleccionados (List[str]): Lista de activos seleccionados
            valor_inicial (float): Valor inicial del portafolio (default: 100)
            
        Returns:
            pd.Series: Serie temporal con la evolución del valor del portafolio
        """
        if self.returns is None:
            self.calcular_rendimientos()
        
        # Filtrar rendimientos solo para activos seleccionados
        returns_selected = self.returns[activos_seleccionados]
        
        # Calcular rendimientos del portafolio
        portfolio_returns = returns_selected.dot(pesos_portafolio)
        
        # Calcular evolución acumulativa
        # Valor_t = Valor_inicial * Π(1 + r_t)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        portfolio_values = valor_inicial * cumulative_returns
        
        return portfolio_values
    
    def get_portfolio_weights_analysis(self, optimized_weights: np.ndarray, selected_assets: List[str], 
                                     all_available_assets: List[str], forced_weights: Dict = None, 
                                     min_weight_threshold: float = 0.001) -> Dict:
        """
        Analiza los pesos del portafolio optimizado y categoriza los activos.
        
        Args:
            optimized_weights (np.ndarray): Pesos optimizados
            selected_assets (List[str]): Lista de activos seleccionados por el usuario
            all_available_assets (List[str]): Lista de todos los activos disponibles
            forced_weights (Dict): Pesos forzados por el usuario
            min_weight_threshold (float): Umbral mínimo para considerar un peso (default: 0.1%)
            
        Returns:
            Dict: Diccionario con análisis completo de pesos
        """
        # Crear diccionario con pesos optimizados
        weights_dict = {}
        
        # Asignar pesos optimizados a activos seleccionados
        for i, asset in enumerate(selected_assets):
            weights_dict[asset] = optimized_weights[i]
        
        # Aplicar pesos forzados si existen
        if forced_weights:
            for asset, forced_weight in forced_weights.items():
                if asset in weights_dict:
                    weights_dict[asset] = forced_weight
        
        # Filtrar pesos por umbral mínimo
        portfolio_final = {}
        considerados_no_utilizados = []
        
        for asset, weight in weights_dict.items():
            if weight >= min_weight_threshold:
                portfolio_final[asset] = weight
            else:
                considerados_no_utilizados.append(asset)
        
        # Renormalizar pesos para que sumen 1
        total_weight = sum(portfolio_final.values())
        if total_weight > 0:
            portfolio_final = {asset: weight/total_weight for asset, weight in portfolio_final.items()}
        
        # Identificar activos excluidos por el usuario
        excluidos_usuario = [asset for asset in all_available_assets if asset not in selected_assets]
        
        # Crear tabla final simplificada con información sobre pesos forzados
        tabla_data = []
        for asset, weight in sorted(portfolio_final.items(), key=lambda x: x[1], reverse=True):
            es_forzado = forced_weights and asset in forced_weights
            tabla_data.append({
                'Activo': asset,
                'Porcentaje': f"{weight:.1%}",
                'Es_Forzado': es_forzado
            })
        
        # Agregar fila de sumatoria
        tabla_data.append({
            'Activo': 'TOTAL',
            'Porcentaje': '100.0%',
            'Es_Forzado': False
        })
        
        tabla_final = pd.DataFrame(tabla_data)
        
        return {
            'tabla_pesos': tabla_final,
            'portfolio_final': portfolio_final,
            'considerados_no_utilizados': considerados_no_utilizados,
            'excluidos_usuario': excluidos_usuario,
            'total_activos_utilizados': len(portfolio_final),
            'umbral_minimo': min_weight_threshold
        }
    
    def get_calculation_summary(self) -> Dict:
        """
        Obtiene un resumen de todos los cálculos realizados.
        FASE 2: Resumen de fórmulas auxiliares implementadas
        
        Returns:
            Dict: Diccionario con resumen de cálculos
        """
        return {
            'fase': 'FASE 2: Backend de Cálculos Cuantitativos',
            'estado': 'Fórmulas auxiliares implementadas (sin optimización)',
            'datos_periodo': {
                'inicio': str(self.data['Date'].min()),
                'fin': str(self.data['Date'].max()),
                'observaciones': len(self.data)
            },
            'activos_analizados': len(self.prices.columns),
            'calculos_disponibles': [
                'Rendimientos históricos mensuales',
                'Rendimiento esperado anualizado (fórmula multiplicativa)',
                'Volatilidad anualizada (fórmula multiplicativa)',
                'Matriz de covarianzas',
                'Matriz de correlaciones',
                'Métricas por activo',
                'Métricas portafolio pesos iguales'
            ],
            'formulas_implementadas': {
                'rendimientos': '(P_t / P_{t-1}) - 1',
                'rendimiento_anual': '(1 + r_mensual)^12 - 1',
                'volatilidad_anual': 'σ_mensual * √12',
                'covarianza_anual': 'Cov_mensual * 12',
                'correlacion': 'Corr_mensual (no cambia)'
            }
        }


def create_quantitative_calculations(data_loader) -> QuantitativeCalculations:
    """
    Función de conveniencia para crear los cálculos cuantitativos.
    FASE 2: Backend de Cálculos Cuantitativos
    
    Args:
        data_loader: Instancia del DataLoader con datos cargados
        
    Returns:
        QuantitativeCalculations: Instancia con cálculos cuantitativos
    """
    if not hasattr(data_loader, 'data') or data_loader.data is None:
        st.error("❌ No hay datos cargados. Primero ejecuta la Funcionalidad 0.")
        return None
    
    # Obtener datos de activos válidos (excluyendo ^GSPC)
    asset_data = data_loader.get_all_asset_data()
    valid_assets = [col for col in asset_data.columns if col != '^GSPC']
    
    if len(valid_assets) == 0:
        st.error("❌ No hay activos válidos para análisis.")
        return None
    
    # Crear DataFrame solo con activos válidos
    analysis_data = asset_data[['Date'] + valid_assets].copy()
    
    # Crear instancia de cálculos cuantitativos
    calculations = QuantitativeCalculations(analysis_data)
    
    return calculations


def mostrar_resumen_calculos(calculations: QuantitativeCalculations):
    """
    Muestra un resumen de los cálculos cuantitativos realizados.
    FASE 2: Visualización de fórmulas auxiliares
    
    Args:
        calculations: Instancia de QuantitativeCalculations
    """
    if calculations is None:
        return
    
    st.subheader("📊 FASE 2: Backend de Cálculos Cuantitativos")
    st.info("ℹ️ **Estado:** Fórmulas auxiliares implementadas (sin optimización)")
    
    # Obtener resumen
    summary = calculations.get_calculation_summary()
    
    # Mostrar información del período
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Período de Análisis", f"{summary['datos_periodo']['observaciones']} meses")
    with col2:
        st.metric("Activos Analizados", summary['activos_analizados'])
    with col3:
        st.metric("Fase del Proyecto", "FASE 2")
    
    # Mostrar fórmulas implementadas
    st.markdown("### 🔢 Fórmulas Implementadas")
    
    formulas = summary['formulas_implementadas']
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Rendimientos:**")
        st.code(f"Rendimiento mensual = {formulas['rendimientos']}")
        st.code(f"Rendimiento anual = {formulas['rendimiento_anual']}")
    
    with col2:
        st.markdown("**Riesgo:**")
        st.code(f"Volatilidad anual = {formulas['volatilidad_anual']}")
        st.code(f"Covarianza anual = {formulas['covarianza_anual']}")
    
    # Mostrar cálculos disponibles
    st.markdown("### ✅ Cálculos Disponibles")
    for calc in summary['calculos_disponibles']:
        st.success(f"✅ {calc}")
    
    st.warning("⚠️ **Recordatorio:** Esta es la FASE 2. Todavía NO se hace optimización de portafolio.")

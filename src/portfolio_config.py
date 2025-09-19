"""
Módulo para manejar la configuración del portafolio de inversión.
Funcionalidad 1: Inputs y Configuración Inicial
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import streamlit as st


class PortfolioConfig:
    """
    Clase para manejar la configuración del portafolio de inversión.
    """
    
    def __init__(self, available_assets: List[str]):
        """
        Inicializa la configuración del portafolio.
        
        Args:
            available_assets (List[str]): Lista de activos disponibles
        """
        self.available_assets = available_assets
        self.selected_assets = available_assets.copy()  # Por defecto, todos seleccionados
        self.forced_weights = {}  # Pesos forzados por el usuario
        self.risk_free_rate_annual = 0.04  # 4% anual por defecto
        self.risk_free_rate_monthly = self.risk_free_rate_annual / 12
        
    def set_selected_assets(self, selected: List[str]):
        """
        Establece los activos seleccionados.
        
        Args:
            selected (List[str]): Lista de activos seleccionados
        """
        self.selected_assets = selected
        
    def set_forced_weight(self, asset: str, weight: float):
        """
        Establece un peso forzado para un activo específico.
        
        Args:
            asset (str): Ticker del activo
            weight (float): Peso (entre 0 y 1)
        """
        if 0 <= weight <= 1:
            self.forced_weights[asset] = weight
        else:
            raise ValueError(f"El peso debe estar entre 0 y 1, recibido: {weight}")
    
    def remove_forced_weight(self, asset: str):
        """
        Remueve un peso forzado para un activo.
        
        Args:
            asset (str): Ticker del activo
        """
        if asset in self.forced_weights:
            del self.forced_weights[asset]
    
    def set_risk_free_rate(self, rate_annual: float):
        """
        Establece la tasa libre de riesgo anual.
        
        Args:
            rate_annual (float): Tasa anual (ej: 0.04 para 4%)
        """
        if rate_annual < 0:
            raise ValueError("La tasa libre de riesgo no puede ser negativa")
        
        self.risk_free_rate_annual = rate_annual
        self.risk_free_rate_monthly = rate_annual / 12
    
    def get_forced_weights_sum(self) -> float:
        """
        Calcula la suma de todos los pesos forzados.
        
        Returns:
            float: Suma de pesos forzados
        """
        return sum(self.forced_weights.values())
    
    def get_remaining_weight(self) -> float:
        """
        Calcula el peso restante disponible para optimización.
        
        Returns:
            float: Peso restante (1 - suma de pesos forzados)
        """
        return 1.0 - self.get_forced_weights_sum()
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """
        Valida la configuración del portafolio.
        
        Returns:
            Tuple[bool, List[str]]: (es_válida, lista_de_errores)
        """
        errors = []
        
        # Validar que hay al menos un activo seleccionado
        if len(self.selected_assets) == 0:
            errors.append("Debe seleccionar al menos un activo")
        
        # Validar que los activos con peso forzado están seleccionados
        for asset in self.forced_weights.keys():
            if asset not in self.selected_assets:
                errors.append(f"El activo {asset} tiene peso forzado pero no está seleccionado")
        
        # Validar que la suma de pesos forzados no exceda 100%
        forced_sum = self.get_forced_weights_sum()
        if forced_sum > 1.0:
            errors.append(f"La suma de pesos forzados ({forced_sum:.1%}) excede el 100%")
        
        # Validar que hay activos disponibles para optimización
        if forced_sum >= 1.0 and len(self.selected_assets) > len(self.forced_weights):
            errors.append("No hay peso disponible para optimizar los activos restantes")
        
        # Validar tasa libre de riesgo
        if self.risk_free_rate_annual < 0:
            errors.append("La tasa libre de riesgo no puede ser negativa")
        
        if self.risk_free_rate_annual > 0.5:  # 50% anual es muy alto
            errors.append("La tasa libre de riesgo parece muy alta (>50% anual)")
        
        return len(errors) == 0, errors
    
    def get_optimization_assets(self) -> List[str]:
        """
        Obtiene los activos que serán optimizados (seleccionados sin peso forzado).
        
        Returns:
            List[str]: Lista de activos para optimización
        """
        return [asset for asset in self.selected_assets if asset not in self.forced_weights]
    
    def get_configuration_summary(self) -> Dict:
        """
        Obtiene un resumen de la configuración actual.
        
        Returns:
            Dict: Diccionario con resumen de la configuración
        """
        return {
            'total_assets': len(self.available_assets),
            'selected_assets': len(self.selected_assets),
            'forced_weights_count': len(self.forced_weights),
            'optimization_assets': len(self.get_optimization_assets()),
            'forced_weights_sum': self.get_forced_weights_sum(),
            'remaining_weight': self.get_remaining_weight(),
            'risk_free_rate_annual': self.risk_free_rate_annual,
            'risk_free_rate_monthly': self.risk_free_rate_monthly,
            'selected_assets_list': self.selected_assets,
            'forced_weights': self.forced_weights.copy(),
            'optimization_assets_list': self.get_optimization_assets()
        }


def create_portfolio_config_interface(data_loader) -> PortfolioConfig:
    """
    Crea la interfaz de configuración del portafolio.
    
    Args:
        data_loader: Instancia del DataLoader con datos cargados
        
    Returns:
        PortfolioConfig: Configuración del portafolio
    """
    if not hasattr(data_loader, 'valid_assets') or not data_loader.valid_assets:
        st.error("❌ No hay datos cargados. Primero ejecuta la Funcionalidad 0.")
        return None
    
    # Obtener activos disponibles (excluyendo ^GSPC que es solo benchmark)
    available_assets = [asset for asset in data_loader.valid_assets if asset != '^GSPC']
    
    # Inicializar configuración
    config = PortfolioConfig(available_assets)
    
    # Título de la sección
    st.subheader("🎯 Configuración del Portafolio")
    
    # 1. Selección de Activos
    st.markdown("### 📋 1. Selección de Activos")
    st.markdown("Selecciona los activos que quieres incluir en tu portafolio:")
    
    # Selección de activos usando multiselect (más simple y sin problemas de CSS)
    selected_assets = st.multiselect(
        "Activos disponibles:",
        options=available_assets,
        default=available_assets,  # Todos seleccionados por defecto
        key="asset_selection",
        help="Selecciona los activos que quieres incluir en tu portafolio"
    )

    # Actualizar activos seleccionados
    config.set_selected_assets(selected_assets)
    # 2. Pesos Forzados
    st.markdown("### ⚖️ 2. Pesos Forzados (Opcional)")
    st.markdown("Asigna pesos específicos a activos particulares. Los activos restantes se optimizarán automáticamente.")
    
    # Solo mostrar activos seleccionados para pesos forzados
    if selected_assets:
        st.markdown("**Activos seleccionados disponibles para peso forzado:**")
        
        # Crear interfaz para pesos forzados
        forced_weights = {}
        
        for asset in selected_assets:
            # Layout mejorado: ticker y input en la misma línea
            st.markdown(f"**{asset}** (porcentaje)")
            
            # Input numérico para peso (en porcentaje) con ancho reducido
            col1, col2, col3 = st.columns([1, 1, 2])  # 25% del ancho para el input
            with col1:
                weight_input = st.number_input(
                    f"Peso para {asset} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.1,
                    format="%.1f",
                    key=f"weight_{asset}"
                )
            
            if weight_input > 0:
                forced_weights[asset] = weight_input / 100.0  # Convertir a decimal
        
        # Actualizar pesos forzados
        for asset, weight in forced_weights.items():
            config.set_forced_weight(asset, weight)
        
        # Mostrar resumen de pesos forzados
        if forced_weights:
            st.markdown("**Pesos forzados asignados:**")
            for asset, weight in forced_weights.items():
                st.info(f"• **{asset}**: {weight:.1%}")
            
            total_forced = sum(forced_weights.values())
            remaining = 1.0 - total_forced
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Peso Forzado Total", f"{total_forced:.1%}")
            with col2:
                st.metric("Peso Disponible para Optimización", f"{remaining:.1%}")
        else:
            st.info("ℹ️ No hay pesos forzados. Todos los activos seleccionados se optimizarán automáticamente.")
    
    st.markdown("---")
    
    # 3. Tasa Libre de Riesgo
    st.markdown("### 🏦 3. Tasa Libre de Riesgo")
    st.markdown("Define la tasa libre de riesgo para los cálculos del modelo.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input para tasa libre de riesgo
        risk_free_input = st.number_input(
            "Tasa Libre de Riesgo Anual (%)",
            min_value=0.0,
            max_value=50.0,
            value=4.0,  # 4% por defecto
            step=0.1,
            format="%.2f",
            help="Tasa de referencia sin riesgo (ej: Treasury a 10 años)"
        )
    
    with col2:
        # Mostrar conversión mensual con método compuesto
        monthly_rate = ((1 + risk_free_input / 100) ** (1/12) - 1) * 100
        st.metric(
            "Tasa Mensual", 
            f"{monthly_rate:.4f}%",
            help="La tasa mensual se calcula con el método compuesto: (1 + tasa anual)^(1/12) - 1"
        )
    
    # Actualizar tasa libre de riesgo
    config.set_risk_free_rate(risk_free_input / 100.0)  # Convertir a decimal
    
    st.markdown("---")
    
    # 4. Validación y Resumen
    st.markdown("### ✅ 4. Validación de Configuración")
    
    # Validar configuración
    is_valid, errors = config.validate_configuration()
    
    if is_valid:
        st.success("✅ **Configuración válida** - Lista para optimización")
        
        # Mostrar resumen final
        summary = config.get_configuration_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Activos Seleccionados", summary['selected_assets'])
        
        with col2:
            st.metric("Pesos Forzados", summary['forced_weights_count'])
        
        with col3:
            st.metric("Para Optimización", summary['optimization_assets'])
        
        with col4:
            st.metric("Tasa Libre Riesgo", f"{summary['risk_free_rate_annual']:.1%}")
        
        # Botón para confirmar configuración
        if st.button("🎯 Confirmar Configuración", type="primary"):
            # Guardar en session state
            st.session_state['portfolio_config'] = config
            st.session_state['config_confirmed'] = True
            
            st.success("🎉 **Configuración confirmada exitosamente!**")
            st.info("💡 **Próximo paso:** Procede a la Funcionalidad 2 para el análisis de riesgos y dependencias.")
            
            # Mostrar detalles de la configuración confirmada
            with st.expander("📋 Ver Detalles de la Configuración"):
                st.json(summary)
    
    else:
        st.error("❌ **Configuración inválida** - Corrige los siguientes errores:")
        for error in errors:
            st.error(f"• {error}")
    
    return config

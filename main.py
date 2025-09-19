"""
Aplicación principal del Portafolio de Inversión con Optimización de Markowitz
Desarrollado con Python puro (NumPy/SciPy) sin dependencias externas como PyPortfolioOpt
"""

import streamlit as st
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import DataLoader
from theme import apply_dark_theme, get_color_scheme
from portfolio_config import create_portfolio_config_interface
from quantitative_calculations import create_quantitative_calculations, mostrar_resumen_calculos

# Configuración de la página
st.set_page_config(
    page_title="Portafolio de Inversión - Optimización Markowitz",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aplicar tema oscuro
apply_dark_theme()

def main():
    """
    Función principal de la aplicación.
    """
    # Separador inicial
    st.markdown("---")
    
    # Sidebar con navegación
    st.sidebar.title("🧭 Navegación")
    
    # Opciones del menú
    menu_options = [
        "🏠 Inicio",
        "📈 Carga y Preparación de Datos",
        "⚙️ Inputs y Configuración Inicial",
        "🎯 Optimización y Frontera Eficiente",
        "📋 Resultados Finales y Validación Histórica"
    ]
    
    selected_page = st.sidebar.selectbox(
        "Selecciona una funcionalidad:",
        menu_options
    )
    
    # Navegación entre páginas
    if selected_page == "🏠 Inicio":
        show_home_page()
    elif selected_page == "📈 Carga y Preparación de Datos":
        show_functionality_0()
    elif selected_page == "⚙️ Inputs y Configuración Inicial":
        show_functionality_1()
    elif selected_page == "🎯 Optimización y Frontera Eficiente":
        show_functionality_2()
    elif selected_page == "📋 Resultados Finales y Validación Histórica":
        show_functionality_3()


def show_home_page():
    """
    Muestra la página de inicio con información del aplicativo para el usuario.
    """
    st.header("📊 Aplicativo para Optimizar Portafolios de Inversión con Markowitz")
    
    st.markdown("### 🎯 ¿Qué hace este aplicativo?")
    
    st.markdown("""
    Este aplicativo carga precios históricos de 20 activos más el índice S&P 500 y permite al usuario seleccionar 
    cuáles activos desea incluir en su análisis. Además, ofrece la opción de asignar pesos específicos a ciertos 
    activos y luego ejecutar la optimización del portafolio siguiendo la teoría de Markowitz.
    """)
    
    st.markdown("### 📋 Páginas del Aplicativo")
    
    st.markdown("""
    El aplicativo está organizado en cuatro páginas principales, cada una con un propósito específico:
    """)
    
    # Información de cada página
    pages_info = [
        {
            "icon": "📈",
            "title": "Carga y Preparación de Datos",
            "description": "Permite cargar y validar los datos históricos de los 20 activos y del índice S&P 500. El sistema verifica automáticamente la integridad de los datos y excluye aquellos activos que presenten información incompleta."
        },
        {
            "icon": "⚙️", 
            "title": "Inputs y Configuración Inicial",
            "description": "Aquí el usuario define qué activos desea incluir en el análisis y puede asignar pesos específicos si lo desea. También permite configurar la tasa libre de riesgo y otros parámetros del modelo."
        },
        {
            "icon": "🎯",
            "title": "Optimización y Frontera Eficiente", 
            "description": "Presenta un mapa de calor con la matriz de correlaciones, una tabla comparativa de los diferentes portafolios (pesos iguales, optimizado sin restricciones y optimizado con restricciones, si las hubiera) y la gráfica de la frontera eficiente junto con la línea de mercado de capitales."
        },
        {
            "icon": "📋",
            "title": "Resultados Finales y Validación Histórica",
            "description": "Muestra los pesos finales del portafolio y un gráfico comparativo entre tres escenarios: portafolio de pesos iguales, portafolio optimizado (con restricciones si las hubiera) y el benchmark del S&P 500, simulando una inversión de $100 hace cinco años."
        }
    ]
    
    for page in pages_info:
        with st.expander(f"{page['icon']} {page['title']}"):
            st.markdown(page['description'])
    
    st.markdown("### 🚀 Comenzar")
    st.info("💡 **Recomendación:** Comienza con la **Carga y Preparación de Datos** para cargar y validar los datos del portafolio.")


def show_functionality_0():
    """
    Implementa la Funcionalidad 0: Carga y Preparación de Datos
    """
    st.header("📈 Carga y Preparación de Datos")
    
    # Mensaje de instrucciones para el usuario
    st.markdown("""
    <div style="color: white; background-color: #1a1a1a; padding: 20px; border-radius: 10px; border: 1px solid #333;">
    
    🔹 <strong>Opciones de Carga de Datos</strong><br><br>
    
    Tienes dos formas de cargar los datos:<br><br>
    
    <strong>1. Usar el archivo por defecto</strong><br>
    • Este archivo ya viene con la aplicación.<br>
    • Contiene precios mensuales de <strong>20 empresas</strong> más el <strong>índice S&P 500</strong>.<br><br>
    
    <strong>2. Subir un archivo CSV personalizado</strong><br>
    • La <strong>primera columna</strong> debe contener las <strong>fechas</strong>.<br>
    • Las <strong>20 columnas siguientes</strong> deben contener los <strong>precios mensuales de 20 empresas</strong>.<br>
    • La <strong>última columna</strong> debe corresponder al <strong>índice S&P 500</strong>.<br>
    • El archivo debe tener exactamente <strong>60 registros mensuales</strong>.<br>
    
    </div>
    """, unsafe_allow_html=True)
    
    # Opciones de carga de datos
    st.markdown('<div style="color: white;"><h3>📂 Selecciona tu Opción</h3></div>', unsafe_allow_html=True)
    
    # File uploader para archivos personalizados
    st.markdown('<div style="color: white;">📁 <strong>Cargar archivo CSV personalizado</strong></div>', unsafe_allow_html=True)
    
    # CSS personalizado para mejorar el contraste del file uploader
    st.markdown("""
    <style>
    /* Estilo para el botón Browse Files - fondo blanco, texto negro */
    .stFileUploader button {
        background-color: white !important;
        color: black !important;
        border: 2px solid #666 !important;
        font-weight: bold !important;
        border-radius: 6px !important;
    }
    
    /* Estilo para el botón Browse Files en hover */
    .stFileUploader button:hover {
        background-color: #f0f0f0 !important;
        color: black !important;
        border: 2px solid #333 !important;
    }
    
    /* Estilo para mejorar visibilidad del área de drag and drop */
    .stFileUploader > div > div {
        background-color: #f8f9fa !important;
        border: 2px dashed #666 !important;
        border-radius: 10px !important;
    }
    
    /* Texto del drag and drop en negro para contraste */
    .stFileUploader div[data-testid="stFileUploaderDropzone"] {
        color: #333 !important;
        font-weight: bold !important;
    }
    
    /* Texto pequeño del límite de tamaño */
    .stFileUploader small {
        color: #666 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Cambiar el texto del label a blanco
    st.markdown('<div style="color: white; font-weight: bold; margin-bottom: 5px;">Selecciona tu archivo CSV</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        label="archivo_csv",
        type=['csv'],
        help="Archivo CSV con primera columna 'Date', 20 columnas de empresas y última columna S&P 500",
        label_visibility="collapsed"
    )
    
    # Botones para cargar datos
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Cargar Archivo Personalizado", type="primary", disabled=uploaded_file is None):
            if uploaded_file is not None:
                with st.spinner("Cargando y validando archivo personalizado..."):
                    try:
                        # Cargar datos desde archivo cargado
                        data_loader = DataLoader()
                        
                        if data_loader.load_data_from_uploaded_file(uploaded_file):
                            # Obtener resumen
                            summary = data_loader.get_data_summary()
                            
                            # Mostrar resultados
                            st.success("✅ Carga de archivo personalizado exitosa!")
                            
                            # Mostrar información del origen
                            st.info(f"📂 **Origen:** Archivo cargado por el usuario: `{uploaded_file.name}`")
                            
                            # Mostrar métricas y detalles (reutilizar lógica)
                            _display_data_summary(summary, data_loader)
                            
                        else:
                            st.error("❌ Error al procesar el archivo personalizado. Verifica el formato y estructura.")
                            
                    except Exception as e:
                        st.error(f"❌ Error inesperado al procesar el archivo: {str(e)}")
    
    with col2:
        if st.button("📋 Usar Archivo por Defecto", type="secondary"):
            with st.spinner("Cargando y validando datos por defecto..."):
                try:
                    # Cargar datos desde archivo por defecto
                    data_loader = DataLoader()
                    
                    if data_loader.load_data():
                        # Obtener resumen
                        summary = data_loader.get_data_summary()
                        
                        # Mostrar resultados
                        st.success("✅ Carga de datos por defecto exitosa!")
                        
                        # Mostrar información del origen
                        st.info("📂 **Origen:** Archivo por defecto del sistema")
                        
                        # Mostrar métricas y detalles (reutilizar lógica)
                        _display_data_summary(summary, data_loader)
                        
                    else:
                        st.error("❌ Error al cargar los datos por defecto. Verifica que el archivo existe.")
                        
                except Exception as e:
                    st.error(f"❌ Error inesperado: {str(e)}")
    
    # Mostrar estado actual
    if 'data_loaded' in st.session_state and st.session_state['data_loaded']:
        data_loader = st.session_state['data_loader']
        summary = data_loader.get_data_summary()
        source_text = "archivo cargado por usuario" if summary.get('data_source') == 'uploaded' else "archivo por defecto"
        st.success(f"✅ Datos cargados y listos para análisis (desde {source_text})")


def _display_data_summary(summary, data_loader):
    """
    Función auxiliar para mostrar el resumen de datos y guardar en session state
    """
    # Información del dataset
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Activos Válidos", summary['total_assets'])
    
    with col2:
        st.metric("Activos Excluidos", summary['excluded_assets'])
    
    with col3:
        st.metric("Observaciones", summary['observations'])
    
    with col4:
        st.metric("Período", f"{summary['date_range']['start']} a {summary['date_range']['end']}")
    
    # Detalles de activos válidos
    st.markdown('<div style="color: white;"><h3>📋 Activos Disponibles</h3></div>', unsafe_allow_html=True)
    
    if summary['valid_assets']:
        # Mostrar en columnas
        cols = st.columns(3)
        for i, asset in enumerate(summary['valid_assets']):
            with cols[i % 3]:
                st.success(f"✅ {asset}")
    
    # Activos excluidos (si los hay)
    if summary['excluded_assets_list']:
        st.markdown('<div style="color: white;"><h3>⚠️ Activos Excluidos</h3></div>', unsafe_allow_html=True)
        for asset in summary['excluded_assets_list']:
            st.warning(f"❌ {asset}")
    
    # Guardar en session state para uso en otras funcionalidades
    st.session_state['data_loader'] = data_loader
    st.session_state['data_loaded'] = True
    
    # Mostrar preview de datos
    st.markdown('<div style="color: white;"><h3>👀 Vista Previa de los Datos</h3></div>', unsafe_allow_html=True)
    
    # Mostrar primeras filas
    preview_data = data_loader.get_all_asset_data().head(10)
    st.dataframe(preview_data, width='stretch')
    
    # Información adicional
    st.info("""
    💡 **Próximo paso:** Los datos han sido validados exitosamente. 
    Puedes proceder a **Inputs y Configuración Inicial** para configurar los parámetros del modelo.
    """)


def show_functionality_1():
    """
    Implementa la Funcionalidad 1: Inputs y Configuración Inicial
    """
    st.header("⚙️ Variables de Entrada")
    
    st.markdown("""
    **Objetivo:** Definir el universo de inversión y los supuestos básicos del modelo.
    """)
    
    # Verificar que los datos están cargados
    if 'data_loaded' not in st.session_state or not st.session_state['data_loaded']:
        st.error("❌ **Error:** Primero debes cargar los datos en la Funcionalidad 0.")
        st.info("💡 Ve a la **Carga y Preparación de Datos** y haz clic en 'Cargar y Validar Datos'.")
        return
    
    # Obtener el data loader
    data_loader = st.session_state.get('data_loader')
    if data_loader is None:
        st.error("❌ **Error:** No se encontraron los datos cargados.")
        st.info("💡 Ve a la **Carga y Preparación de Datos** y recarga los datos.")
        return
    
    # Mostrar información de los datos cargados
    st.markdown("### 📊 Datos Cargados")
    summary = data_loader.get_data_summary()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # Mostrar como "20 + S&P500" (total_assets incluye S&P500, así que restamos 1)
        activos_empresas = summary['total_assets'] - 1
        st.metric("Activos Disponibles", f"{activos_empresas} + S&P500")
    with col2:
        st.metric("Observaciones", summary['observations'])
    with col3:
        st.metric("Período", f"{summary['date_range']['start']} a {summary['date_range']['end']}")
    
    st.markdown("---")
    
    # Crear interfaz de configuración
    config = create_portfolio_config_interface(data_loader)
    
    # Mostrar estado de la configuración
    if 'config_confirmed' in st.session_state and st.session_state['config_confirmed']:
        st.markdown("---")
        st.success("🎉 **Configuración confirmada y lista para análisis**")
        
        # Generar cálculos cuantitativos automáticamente en el backend (sin mostrar UI)
        if 'calculations_ready' not in st.session_state or not st.session_state['calculations_ready']:
            with st.spinner("Generando cálculos cuantitativos en el backend..."):
                try:
                    # Crear instancia de cálculos cuantitativos automáticamente
                    calculations = create_quantitative_calculations(data_loader)
                    
                    if calculations is not None:
                        # Guardar en session state (backend)
                        st.session_state['quantitative_calculations'] = calculations
                        st.session_state['calculations_ready'] = True
                        st.success("✅ Cálculos cuantitativos generados automáticamente en el backend")
                    else:
                        st.error("❌ Error al generar los cálculos cuantitativos en el backend.")
                        
                except Exception as e:
                    st.error(f"❌ Error inesperado: {str(e)}")
        else:
            st.success("✅ Cálculos cuantitativos listos en el backend")
        
        # Información para el siguiente paso
        st.info("""
        💡 **Próximo paso:** Los cálculos cuantitativos están listos en el backend. 
        Puedes proceder a **Optimización y Frontera Eficiente** para visualizar la matriz de correlaciones.
        """)
        
        # Botón para modificar configuración
        if st.button("🔄 Modificar Configuración"):
            # Limpiar configuración confirmada
            if 'portfolio_config' in st.session_state:
                del st.session_state['portfolio_config']
            if 'config_confirmed' in st.session_state:
                del st.session_state['config_confirmed']
            # También limpiar cálculos cuantitativos
            if 'quantitative_calculations' in st.session_state:
                del st.session_state['quantitative_calculations']
            if 'calculations_ready' in st.session_state:
                del st.session_state['calculations_ready']
            st.rerun()


def show_functionality_2():
    """
    Implementa la Funcionalidad 2: Optimización y Frontera Eficiente
    """
    st.header("Matriz de Correlaciones")
    
    # Verificar que los datos están cargados
    if 'data_loaded' not in st.session_state or not st.session_state['data_loaded']:
        st.error("❌ **Error:** Primero debes cargar los datos en la Funcionalidad 0.")
        st.info("💡 Ve a la **Carga y Preparación de Datos** y haz clic en 'Cargar y Validar Datos'.")
        return
    
    # Verificar que la configuración está confirmada
    if 'config_confirmed' not in st.session_state or not st.session_state['config_confirmed']:
        st.error("❌ **Error:** Primero debes configurar los parámetros en la Funcionalidad 1.")
        st.info("💡 Ve a **Inputs y Configuración Inicial** y confirma la configuración del portafolio.")
        return
    
    # Verificar que los cálculos cuantitativos están listos
    if 'calculations_ready' not in st.session_state or not st.session_state['calculations_ready']:
        st.error("❌ **Error:** Primero debes generar los cálculos cuantitativos en la Funcionalidad 1.")
        st.info("💡 Ve a **Inputs y Configuración Inicial** y haz clic en 'Generar Cálculos Cuantitativos'.")
        return
    
    # Obtener los cálculos cuantitativos y configuración
    calculations = st.session_state.get('quantitative_calculations')
    portfolio_config = st.session_state.get('portfolio_config')
    
    if calculations is None:
        st.error("❌ **Error:** No se encontraron los cálculos cuantitativos.")
        return
    
    # Obtener activos seleccionados
    selected_assets = []
    if portfolio_config and hasattr(portfolio_config, 'selected_assets'):
        selected_assets = portfolio_config.selected_assets
    
    # Generar matriz de correlaciones solo para activos seleccionados
    correlation_matrix_full = calculations.calcular_matriz_correlaciones()
    if selected_assets:
        correlation_matrix = correlation_matrix_full.loc[selected_assets, selected_assets]
    else:
        correlation_matrix = correlation_matrix_full
    
    # Crear mapa de calor con Plotly
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale=[
            [0.0, '#FF0000'],  # Rojo para correlaciones negativas
            [0.5, '#000000'],  # Negro para correlación cero
            [1.0, '#0000FF']   # Azul para correlaciones positivas
        ],
        zmin=-1,
        zmax=1,
        text=np.round(correlation_matrix.values, 3),
        texttemplate="%{text}",
        textfont={"size": 11, "color": "#FFFFFF"},
        hoverongaps=False,
        hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>' +
                     'Correlación: %{z:.3f}<br>' +
                     '<extra></extra>',
        showscale=True,
        colorbar=dict(
            title="Correlación",
            title_font={"color": "#FFFFFF"},
            tickfont={"color": "#FFFFFF"}
        )
    ))
    
    # Configurar el layout del gráfico
    fig.update_layout(
        title={
            'text': 'Matriz de Correlaciones entre Activos',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#FFFFFF'}
        },
        xaxis={
            'title': {'text': 'Activos', 'font': {'color': '#FFFFFF', 'size': 14}},
            'tickfont': {'color': '#FFFFFF', 'size': 11},
            'tickangle': 45,
            'showgrid': False,
            'side': 'bottom'
        },
        yaxis={
            'title': {'text': 'Activos', 'font': {'color': '#FFFFFF', 'size': 14}},
            'tickfont': {'color': '#FFFFFF', 'size': 11},
            'showgrid': False
        },
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font={'color': '#FFFFFF'},
        width=1200,
        height=800,
        margin=dict(l=150, r=100, t=120, b=150)
    )
    
    # Mostrar el gráfico
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla comparativa de portafolios
    st.markdown("---")
    st.markdown("### Comparación de Portafolios: Pesos Iguales vs Optimizado")
    
    # Obtener la configuración del portafolio
    tasa_libre_riesgo = 0.04  # Default
    forced_weights = {}
    
    if portfolio_config:
        if hasattr(portfolio_config, 'risk_free_rate'):
            tasa_libre_riesgo = portfolio_config.risk_free_rate
        if hasattr(portfolio_config, 'forced_weights'):
            forced_weights = portfolio_config.forced_weights or {}
    
    # Calcular métricas del portafolio de pesos iguales
    equal_weights_metrics = calculations.calcular_metricas_portafolio_pesos_iguales()
    equal_weights_sharpe = calculations.calcular_sharpe_ratio(
        equal_weights_metrics['rendimiento_esperado_anual'],
        equal_weights_metrics['volatilidad_anual'],
        tasa_libre_riesgo
    )
    
    # Calcular portafolio optimizado
    with st.spinner("Optimizando portafolio..."):
        optimized_portfolio = calculations.optimizar_portafolio_sharpe(tasa_libre_riesgo, selected_assets)
    
    if optimized_portfolio['optimizacion_exitosa']:
        # Crear tabla comparativa (lógica de 2 o 3 columnas)
        comparison_data = {
            'Métrica': [
                'Rendimiento Esperado Anual',
                'Volatilidad Anual (Riesgo)',
                'Ratio de Sharpe'
            ],
            'Portafolio Pesos Iguales': [
                f"{equal_weights_metrics['rendimiento_esperado_anual']:.2%}",
                f"{equal_weights_metrics['volatilidad_anual']:.2%}",
                f"{equal_weights_sharpe:.3f}"
            ],
            'Portafolio Optimizado': [
                f"{optimized_portfolio['rendimiento_esperado_anual']:.2%}",
                f"{optimized_portfolio['volatilidad_anual']:.2%}",
                f"{optimized_portfolio['sharpe_ratio']:.3f}"
            ]
        }
        
        # Si hay pesos forzados, agregar tercera columna
        if forced_weights:
            # Calcular portafolio optimizado con restricciones
            with st.spinner("Calculando portafolio con restricciones..."):
                constrained_portfolio = calculations.optimizar_portafolio_con_restricciones(
                    tasa_libre_riesgo, selected_assets, forced_weights
                )
            
            if constrained_portfolio['optimizacion_exitosa']:
                constrained_sharpe = calculations.calcular_sharpe_ratio(
                    constrained_portfolio['rendimiento_esperado_anual'],
                    constrained_portfolio['volatilidad_anual'],
                    tasa_libre_riesgo
                )
                
                comparison_data['Portafolio Optimizado con Restricciones'] = [
                    f"{constrained_portfolio['rendimiento_esperado_anual']:.2%}",
                    f"{constrained_portfolio['volatilidad_anual']:.2%}",
                    f"{constrained_sharpe:.3f}"
                ]
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Mostrar tabla
        st.dataframe(comparison_df, width='stretch', hide_index=True)
        
        # Nota explicativa debajo de la tabla
        st.markdown("#### 📝 Nota sobre las Métricas")
        
        n_empresas = len(selected_assets)
        peso_igual = 100.0 / n_empresas
        
        nota_text = f"""
        • **Estas métricas** (rendimiento esperado anual, volatilidad anual y razón de Sharpe) se calcularon con base en las {n_empresas} empresas filtradas.
        
        • **El Portafolio de Pesos Iguales** asigna 100% entre {n_empresas} empresas, cada una con {peso_igual:.1f}%.
        
        • **El Portafolio Optimizado** se construye con esas {n_empresas} empresas sin restricción de pesos.
        """
        
        if forced_weights:
            restricciones_text = []
            for asset, weight in forced_weights.items():
                restricciones_text.append(f"{weight:.1%} a {asset}")
            restricciones_str = ", ".join(restricciones_text)
            
            nota_text += f"""
        • **El Portafolio Optimizado con Restricción de Pesos** respeta las asignaciones definidas por el usuario ({restricciones_str}) y optimiza el resto para maximizar la razón de Sharpe.
            """
        
        st.markdown(nota_text)
        
        # Gráfico de Frontera Eficiente
        st.markdown("---")
        st.markdown("### Frontera Eficiente y Línea del Mercado de Capitales")
        
        with st.spinner("Calculando frontera eficiente..."):
            # Calcular frontera eficiente usando solo activos seleccionados
            frontera = calculations.calcular_frontera_eficiente(n_puntos=100, activos_seleccionados=selected_assets)
            
            if frontera['exito']:
                # Calcular CML usando el portafolio optimizado como proxy del mercado
                cml = calculations.calcular_cml(
                    optimized_portfolio['rendimiento_esperado_anual'],
                    optimized_portfolio['volatilidad_anual'],
                    tasa_libre_riesgo,
                    n_puntos=100
                )
                
                # Crear gráfico con Plotly
                fig = go.Figure()
                
                # Agregar frontera eficiente
                fig.add_trace(go.Scatter(
                    x=frontera['volatilidades'],
                    y=frontera['rendimientos'],
                    mode='lines',
                    name='Frontera Eficiente',
                    line=dict(color='#00FFFF', width=3),
                    hovertemplate='<b>Frontera Eficiente</b><br>' +
                                'Volatilidad: %{x:.2%}<br>' +
                                'Rendimiento: %{y:.2%}<br>' +
                                '<extra></extra>'
                ))
                
                # Agregar CML
                fig.add_trace(go.Scatter(
                    x=cml['volatilidades'],
                    y=cml['rendimientos'],
                    mode='lines',
                    name='Línea del Mercado de Capitales (CML)',
                    line=dict(color='#FFD700', width=2, dash='dash'),
                    hovertemplate='<b>CML</b><br>' +
                                'Volatilidad: %{x:.2%}<br>' +
                                'Rendimiento: %{y:.2%}<br>' +
                                '<extra></extra>'
                ))
                
                # Agregar punto del portafolio de pesos iguales
                fig.add_trace(go.Scatter(
                    x=[equal_weights_metrics['volatilidad_anual']],
                    y=[equal_weights_metrics['rendimiento_esperado_anual']],
                    mode='markers',
                    name='Portafolio Pesos Iguales',
                    marker=dict(color='#FF6B6B', size=12, symbol='circle'),
                    hovertemplate='<b>Pesos Iguales</b><br>' +
                                'Volatilidad: %{x:.2%}<br>' +
                                'Rendimiento: %{y:.2%}<br>' +
                                f'Sharpe: {equal_weights_sharpe:.3f}<br>' +
                                '<extra></extra>'
                ))
                
                # Agregar punto del portafolio optimizado (sin restricciones)
                fig.add_trace(go.Scatter(
                    x=[optimized_portfolio['volatilidad_anual']],
                    y=[optimized_portfolio['rendimiento_esperado_anual']],
                    mode='markers',
                    name='Portafolio Optimizado (Max Sharpe)',
                    marker=dict(color='#4ECDC4', size=15, symbol='star'),
                    hovertemplate='<b>Portafolio Optimizado</b><br>' +
                                'Volatilidad: %{x:.2%}<br>' +
                                'Rendimiento: %{y:.2%}<br>' +
                                f'Sharpe: {optimized_portfolio["sharpe_ratio"]:.3f}<br>' +
                                '<extra></extra>'
                ))
                
                # Agregar punto del portafolio con restricciones (solo si existen pesos forzados)
                if forced_weights and 'constrained_portfolio' in locals() and constrained_portfolio['optimizacion_exitosa']:
                    constrained_sharpe_calc = calculations.calcular_sharpe_ratio(
                        constrained_portfolio['rendimiento_esperado_anual'],
                        constrained_portfolio['volatilidad_anual'],
                        tasa_libre_riesgo
                    )
                    
                    fig.add_trace(go.Scatter(
                        x=[constrained_portfolio['volatilidad_anual']],
                        y=[constrained_portfolio['rendimiento_esperado_anual']],
                        mode='markers',
                        name='Portafolio Optimizado con Restricciones',
                        marker=dict(color='#9B59B6', size=12, symbol='diamond'),
                        hovertemplate='<b>Portafolio con Restricciones</b><br>' +
                                    'Volatilidad: %{x:.2%}<br>' +
                                    'Rendimiento: %{y:.2%}<br>' +
                                    f'Sharpe: {constrained_sharpe_calc:.3f}<br>' +
                                    '<extra></extra>'
                    ))
                
                # Agregar punto de la tasa libre de riesgo
                fig.add_trace(go.Scatter(
                    x=[0],
                    y=[tasa_libre_riesgo],
                    mode='markers',
                    name='Tasa Libre de Riesgo',
                    marker=dict(color='#FFFFFF', size=10, symbol='diamond'),
                    hovertemplate='<b>Tasa Libre de Riesgo</b><br>' +
                                'Volatilidad: 0%<br>' +
                                f'Rendimiento: {tasa_libre_riesgo:.2%}<br>' +
                                '<extra></extra>'
                ))
                
                # Configurar el layout del gráfico
                fig.update_layout(
                    title={
                        'text': 'Frontera Eficiente y Línea del Mercado de Capitales',
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 18, 'color': '#FFFFFF'}
                    },
                    xaxis={
                        'title': {'text': 'Volatilidad (Riesgo)', 'font': {'color': '#FFFFFF', 'size': 14}},
                        'tickfont': {'color': '#FFFFFF', 'size': 12},
                        'tickformat': '.1%',
                        'showgrid': True,
                        'gridcolor': '#333333'
                    },
                    yaxis={
                        'title': {'text': 'Rendimiento Esperado', 'font': {'color': '#FFFFFF', 'size': 14}},
                        'tickfont': {'color': '#FFFFFF', 'size': 12},
                        'tickformat': '.1%',
                        'showgrid': True,
                        'gridcolor': '#333333'
                    },
                    plot_bgcolor='#000000',
                    paper_bgcolor='#000000',
                    font={'color': '#FFFFFF'},
                    legend={
                        'bgcolor': 'rgba(0,0,0,0.5)',
                        'bordercolor': '#FFFFFF',
                        'borderwidth': 1,
                        'font': {'color': '#FFFFFF'}
                    },
                    width=1000,
                    height=600,
                    margin=dict(l=80, r=80, t=100, b=80)
                )
                
                # Mostrar el gráfico
                st.plotly_chart(fig, use_container_width=True)
                
                # Información adicional sobre el gráfico
                st.markdown("#### 📖 Interpretación del Gráfico")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div style="background-color: #1f1f1f; padding: 15px; border-radius: 10px; border-left: 4px solid #00FFFF;">
                    <p style="color: white; margin: 0;"><strong>🔵 Frontera Eficiente (Azul):</strong></p>
                    <ul style="color: white; margin: 5px 0;">
                    <li>Representa todos los portafolios óptimos</li>
                    <li>Máximo rendimiento para cada nivel de riesgo</li>
                    <li>Calculada mediante optimización cuadrática</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div style="background-color: #1f1f1f; padding: 15px; border-radius: 10px; border-left: 4px solid #FFD700;">
                    <p style="color: white; margin: 0;"><strong>🟡 CML (Amarilla):</strong></p>
                    <ul style="color: white; margin: 5px 0;">
                    <li>Línea del Mercado de Capitales</li>
                    <li>Combina activo libre de riesgo con portafolio de mercado</li>
                    <li>Pendiente = Ratio de Sharpe del mercado</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
            else:
                st.error("❌ Error al calcular la frontera eficiente.")
    
    else:
        st.error(f"❌ Error en la optimización: {optimized_portfolio.get('error', 'Error desconocido')}")
        st.warning("Se muestra únicamente el portafolio de pesos iguales.")


def show_functionality_3():
    """
    Implementa la Funcionalidad 4: Resultados Finales y Validación Histórica
    """
    
    # Verificar que los datos están cargados
    if 'data_loaded' not in st.session_state or not st.session_state['data_loaded']:
        st.error("❌ **Error:** Primero debes cargar los datos en la Funcionalidad 0.")
        st.info("💡 Ve a la **Carga y Preparación de Datos** y haz clic en 'Cargar y Validar Datos'.")
        return
    
    # Verificar que la configuración está confirmada
    if 'config_confirmed' not in st.session_state or not st.session_state['config_confirmed']:
        st.error("❌ **Error:** Primero debes configurar los parámetros en la Funcionalidad 1.")
        st.info("💡 Ve a **Inputs y Configuración Inicial** y confirma la configuración del portafolio.")
        return
    
    # Verificar que los cálculos cuantitativos están listos
    if 'calculations_ready' not in st.session_state or not st.session_state['calculations_ready']:
        st.error("❌ **Error:** Primero debes generar los cálculos cuantitativos en la Funcionalidad 1.")
        st.info("💡 Ve a **Inputs y Configuración Inicial** y haz clic en 'Generar Cálculos Cuantitativos'.")
        return
    
    # Obtener datos necesarios
    calculations = st.session_state.get('quantitative_calculations')
    portfolio_config = st.session_state.get('portfolio_config')
    data_loader = st.session_state.get('data_loader')
    
    if calculations is None or portfolio_config is None or data_loader is None:
        st.error("❌ **Error:** No se encontraron los datos necesarios.")
        return
    
    # Obtener configuración del portafolio
    selected_assets = portfolio_config.selected_assets
    forced_weights = portfolio_config.forced_weights if hasattr(portfolio_config, 'forced_weights') else {}
    tasa_libre_riesgo = portfolio_config.risk_free_rate if hasattr(portfolio_config, 'risk_free_rate') else 0.04
    
    # Calcular portafolio optimizado (sin restricciones)
    with st.spinner("Calculando portafolio optimizado..."):
        optimized_portfolio = calculations.optimizar_portafolio_sharpe(tasa_libre_riesgo, selected_assets)
    
    if not optimized_portfolio['optimizacion_exitosa']:
        st.error(f"❌ Error en la optimización: {optimized_portfolio.get('error', 'Error desconocido')}")
        return
    
    # Calcular portafolio con restricciones si hay pesos forzados
    constrained_portfolio = None
    if forced_weights:
        with st.spinner("Calculando portafolio con restricciones..."):
            constrained_portfolio = calculations.optimizar_portafolio_con_restricciones(
                tasa_libre_riesgo, selected_assets, forced_weights
            )
    
    # Determinar qué portafolio mostrar en la tabla
    portfolio_to_display = constrained_portfolio if (forced_weights and constrained_portfolio and constrained_portfolio['optimizacion_exitosa']) else optimized_portfolio
    portfolio_title = "Pesos del Portafolio Optimizado con Restricción de Pesos" if (forced_weights and constrained_portfolio and constrained_portfolio['optimizacion_exitosa']) else "Pesos del Portafolio Optimizado"
    
    # 1. Tabla de pesos del portafolio
    st.markdown(f"### 📊 {portfolio_title}")
    
    # Obtener todos los activos disponibles
    all_available_assets = data_loader.valid_assets
    
    # Crear análisis de pesos usando el portafolio correcto
    weights_analysis = calculations.get_portfolio_weights_analysis(
        portfolio_to_display['pesos'],
        selected_assets,
        all_available_assets,
        forced_weights,
        min_weight_threshold=0.001  # 0.1%
    )
    
    # Mostrar tabla simplificada con colores para pesos forzados
    tabla_pesos = weights_analysis['tabla_pesos']
    
    # Crear tabla con colores diferenciados
    if 'Es_Forzado' in tabla_pesos.columns:
        # Crear copia de la tabla sin la columna Es_Forzado para mostrar
        tabla_display = tabla_pesos[['Activo', 'Porcentaje']].copy()
        
        # Aplicar colores a los activos con restricciones
        for idx, row in tabla_pesos.iterrows():
            if row.get('Es_Forzado', False) and forced_weights and row['Activo'] in forced_weights:
                # Marcar visualmente los pesos forzados
                tabla_display.loc[idx, 'Activo'] = f"🔴 {row['Activo']}"
        
        st.dataframe(tabla_display, width='stretch', hide_index=True)
    else:
        # Fallback: mostrar tabla normal
        st.dataframe(tabla_pesos, width='stretch', hide_index=True)
    
    # Nota explicativa sobre restricciones
    if forced_weights and constrained_portfolio and constrained_portfolio['optimizacion_exitosa']:
        st.markdown("#### 📝 Nota sobre Restricciones de Peso")
        
        num_restricciones = len(forced_weights)
        
        if num_restricciones == 1:
            # Una sola restricción
            asset, weight = list(forced_weights.items())[0]
            st.info(f"ℹ️ **{asset}** obtuvo un {weight:.1%} porque era su restricción de peso definida por el usuario.")
        else:
            # Múltiples restricciones
            assets = list(forced_weights.keys())
            weights = list(forced_weights.values())
            
            # Crear lista de activos separados por comas
            if num_restricciones == 2:
                assets_str = f"**{assets[0]}** y **{assets[1]}**"
            else:
                assets_str = ", ".join([f"**{asset}**" for asset in assets[:-1]]) + f" y **{assets[-1]}**"
            
            # Crear lista de porcentajes
            weights_str = ", ".join([f"{weight:.1%}" for weight in weights[:-1]]) + f" y {weights[-1]:.1%}"
            
            st.info(f"ℹ️ {assets_str} obtuvieron {weights_str} respectivamente de acuerdo a las restricciones definidas por el usuario.")
    
    
    # Mostrar información adicional en dos columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚠️ Activos Considerados pero No Seleccionados en el Portafolio Óptimo")
        
        if weights_analysis['considerados_no_utilizados']:
            activos_str = ", ".join(weights_analysis['considerados_no_utilizados'])
            st.markdown(f"{activos_str}")
        else:
            st.markdown("*Ninguno*")
    
    with col2:
        st.markdown("#### ❌ Activos Excluidos por el Usuario")
        
        # Filtrar ^GSPC del listado de excluidos
        excluidos_filtrados = [asset for asset in weights_analysis['excluidos_usuario'] if asset != '^GSPC']
        
        if excluidos_filtrados:
            activos_str = ", ".join(excluidos_filtrados)
            st.markdown(f"{activos_str}")
        else:
            st.markdown("*Ninguno*")
    
    
    # 2. Gráfico de evolución histórica
    st.markdown("---")
    st.markdown("### 📈 Evolución Histórica: $100 Invertidos Hace 5 Años")
    
    with st.spinner("Calculando evolución histórica..."):
        # Calcular pesos iguales solo para activos seleccionados
        n_selected = len(selected_assets)
        equal_weights = np.array([1/n_selected] * n_selected)
        
        # Calcular evolución del portafolio de pesos iguales
        equal_weights_evolution = calculations.calcular_evolucion_historica(
            equal_weights, selected_assets, 100.0
        )
        
        # Calcular evolución del portafolio optimizado (usando el portafolio correcto)
        optimized_evolution = calculations.calcular_evolucion_historica(
            portfolio_to_display['pesos'], selected_assets, 100.0
        )
        
        # Obtener datos del S&P 500 como benchmark
        sp500_data = data_loader.get_all_asset_data()
        if '^GSPC' in sp500_data.columns:
            sp500_prices = sp500_data['^GSPC'].dropna()
            sp500_returns = sp500_prices.pct_change().dropna()
            sp500_evolution = 100.0 * (1 + sp500_returns).cumprod()
        else:
            sp500_evolution = None
        
        # Crear fechas para el eje X (usar índices simples para evitar problemas con fechas)
        dates = list(range(len(equal_weights_evolution)))
        
        # Crear gráfico con Plotly
        fig = go.Figure()
        
        # Agregar línea del portafolio de pesos iguales
        fig.add_trace(go.Scatter(
            x=dates,
            y=equal_weights_evolution,
            mode='lines',
            name='Portafolio Pesos Iguales',
            line=dict(color='#FF6B6B', width=2),
            hovertemplate='<b>Pesos Iguales</b><br>' +
                        'Fecha: %{x}<br>' +
                        'Valor: $%{y:.2f}<br>' +
                        '<extra></extra>'
        ))
        
        # Agregar línea del portafolio optimizado (determinar nombre según si hay restricciones)
        optimized_name = 'Portafolio Optimizado con Restricciones' if (forced_weights and constrained_portfolio and constrained_portfolio['optimizacion_exitosa']) else 'Portafolio Optimizado'
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=optimized_evolution,
            mode='lines',
            name=optimized_name,
            line=dict(color='#4ECDC4', width=2),
            hovertemplate=f'<b>{optimized_name}</b><br>' +
                        'Fecha: %{x}<br>' +
                        'Valor: $%{y:.2f}<br>' +
                        '<extra></extra>'
        ))
        
        # Agregar línea del S&P 500 si está disponible
        if sp500_evolution is not None:
            fig.add_trace(go.Scatter(
                x=dates,
                y=sp500_evolution,
                mode='lines',
                name='S&P 500 (Benchmark)',
                line=dict(color='#FFD700', width=2, dash='dot'),
                hovertemplate='<b>S&P 500</b><br>' +
                            'Fecha: %{x}<br>' +
                            'Valor: $%{y:.2f}<br>' +
                            '<extra></extra>'
            ))
        
        # Configurar el layout del gráfico
        fig.update_layout(
            title={
                'text': 'Evolución Histórica: $100 Invertidos',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#FFFFFF'}
            },
            xaxis={
                'title': {'text': 'Períodos (Meses)', 'font': {'color': '#FFFFFF', 'size': 14}},
                'tickfont': {'color': '#FFFFFF', 'size': 12},
                'showgrid': True,
                'gridcolor': '#333333'
            },
            yaxis={
                'title': {'text': 'Valor Acumulado ($)', 'font': {'color': '#FFFFFF', 'size': 14}},
                'tickfont': {'color': '#FFFFFF', 'size': 12},
                'tickformat': '$,.0f',
                'showgrid': True,
                'gridcolor': '#333333'
            },
            plot_bgcolor='#000000',
            paper_bgcolor='#000000',
            font={'color': '#FFFFFF'},
            legend={
                'bgcolor': 'rgba(0,0,0,0.5)',
                'bordercolor': '#FFFFFF',
                'borderwidth': 1,
                'font': {'color': '#FFFFFF'}
            },
            width=1200,
            height=600,
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        # Mostrar el gráfico
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar valores finales
        st.markdown("#### 💰 Valores Finales")
        
        # Pregunta que responde el gráfico (movida aquí)
        st.markdown("""
        <div style="background-color: #1e3a8a; padding: 15px; border-radius: 10px; margin: 20px 0;">
        <p style="color: white; margin: 0; font-size: 16px;">
        💡 <strong>Este gráfico responde a la pregunta:</strong><br>
        <em>"Si hubieras invertido $100 hace 5 años, ¿cuánto tendrías hoy en cada opción 
        (pesos iguales, portafolio optimizado y S&P 500 como benchmark)?"</em>
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            final_equal = equal_weights_evolution.iloc[-1]
            st.markdown(f"<h4 style='color: white; margin-bottom: 0;'>Pesos Iguales</h4>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color: white; margin-top: 0;'>${final_equal:.2f}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #00ff00; font-size: 14px;'>+${final_equal - 100:.2f}</p>", unsafe_allow_html=True)
        
        with col2:
            final_optimized = optimized_evolution.iloc[-1]
            optimized_title = 'Portafolio Optimizado con Restricciones' if (forced_weights and constrained_portfolio and constrained_portfolio['optimizacion_exitosa']) else 'Portafolio Optimizado'
            st.markdown(f"<h4 style='color: white; margin-bottom: 0;'>{optimized_title}</h4>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color: white; margin-top: 0;'>${final_optimized:.2f}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #00ff00; font-size: 14px;'>+${final_optimized - 100:.2f}</p>", unsafe_allow_html=True)
        
        with col3:
            if sp500_evolution is not None:
                final_sp500 = sp500_evolution.iloc[-1]
                st.markdown(f"<h4 style='color: white; margin-bottom: 0;'>S&P 500</h4>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='color: white; margin-top: 0;'>${final_sp500:.2f}</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #00ff00; font-size: 14px;'>+${final_sp500 - 100:.2f}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h4 style='color: white; margin-bottom: 0;'>S&P 500</h4>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='color: white; margin-top: 0;'>No disponible</h2>", unsafe_allow_html=True)
        
        # Nota explicativa
        st.markdown("---")
        st.markdown("#### 📝 Nota Explicativa")
        
        # Información sobre los portafolios
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🟢 Portafolio de Pesos Iguales:**")
            peso_igual = 100.0 / len(selected_assets)
            n_empresas = len(selected_assets)
            st.markdown(f"Se les asignó a cada una de las siguientes {n_empresas} empresas un peso igual al 100% dividido entre {n_empresas}, equivalente a {peso_igual:.1f}% cada una:")
            st.markdown(", ".join(selected_assets))
        
        with col2:
            st.markdown("**🎯 Portafolio Optimizado:**")
            # Usar solo activos con peso > 0.1%
            activos_utilizados = []
            for asset in selected_assets:
                if asset in weights_analysis['portfolio_final']:
                    peso = weights_analysis['portfolio_final'][asset]
                    activos_utilizados.append(f"{asset}: {peso:.1%}")
            st.markdown(", ".join(activos_utilizados))
        


if __name__ == "__main__":
    main()

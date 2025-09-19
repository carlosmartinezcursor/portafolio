"""
Configuración del tema oscuro para la aplicación Streamlit.
"""

import streamlit as st


def apply_dark_theme():
    """
    Aplica el tema oscuro personalizado a la aplicación Streamlit.
    """
    st.markdown("""
    <style>
    /* Configuración general del tema oscuro */
    .stApp {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* Forzar fondo negro en toda la aplicación */
    .main .block-container {
        background-color: #000000 !important;
    }
    
    /* Sidebar completamente negro */
    .css-1d391kg, .css-1d391kg .css-1v0mbdj, [data-testid="stSidebar"] {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* Header/toolbar negro */
    .css-18e3th9, .css-1d391kg, header[data-testid="stHeader"] {
        background-color: #000000 !important;
    }
    
    /* Forzar todos los contenedores a fondo negro */
    .css-1d391kg, .css-1v0mbdj, .css-k1vhr4, .css-1vencpc {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* Asegurar que el sidebar sea completamente negro */
    section[data-testid="stSidebar"] {
        background-color: #000000 !important;
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: #000000 !important;
    }
    
    /* Header y toolbar completamente negros */
    .css-18e3th9, .css-hxt7ib, .css-1d391kg {
        background-color: #000000 !important;
    }
    
    /* Forzar fondo negro en el área principal */
    .main, .main > div, .block-container {
        background-color: #000000 !important;
    }
    
    /* Específicamente para elementos de Streamlit - más selectivo */
    .stApp, .stApp > div:first-child {
        background-color: #000000 !important;
    }
    
    /* Títulos y texto */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    .stMarkdown {
        color: #ffffff;
    }
    
    /* Widgets */
    .stSelectbox > div > div {
        background-color: #2a2a2a;
        color: #ffffff;
    }
    
    .stTextInput > div > div > input {
        background-color: #2a2a2a;
        color: #ffffff;
        border: 1px solid #444444;
    }
    
    .stNumberInput > div > div > input {
        background-color: #2a2a2a;
        color: #ffffff;
        border: 1px solid #444444;
    }
    
    /* Botones */
    .stButton > button {
        background-color: #0066cc;
        color: #ffffff;
        border: none;
        border-radius: 5px;
    }
    
    .stButton > button:hover {
        background-color: #0052a3;
    }
    
    /* Tablas - mantener visibilidad */
    .stDataFrame {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    /* Asegurar que las tablas sean visibles */
    .stDataFrame table, .stDataFrame thead, .stDataFrame tbody, .stDataFrame tr, .stDataFrame td, .stDataFrame th {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border-color: #444444 !important;
    }
    
    /* Gráficos de Plotly - mantener visibilidad */
    .plotly, .plotly-graph-div, .js-plotly-plot {
        background-color: transparent !important;
    }
    
    /* Métricas */
    .metric-container {
        background-color: #1a1a1a;
        border: 1px solid #333333;
        border-radius: 5px;
        padding: 10px;
    }
    
    /* Alertas */
    .stAlert {
        background-color: #2a2a2a;
        border: 1px solid #444444;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #2a2a2a;
        color: #ffffff;
    }
    
    /* Checkbox */
    .stCheckbox > label > div {
        background-color: transparent;
    }
    
    /* Mejorar legibilidad de checkboxes */
    .stCheckbox > label {
        color: #ffd700 !important;
        font-size: 16px !important;
        font-weight: bold !important;
    }
    
    .stCheckbox > label > div[data-testid="stMarkdownContainer"] {
        color: #ffd700 !important;
    }
    
    .stCheckbox > label > div[data-testid="stMarkdownContainer"] > p {
        color: #ffd700 !important;
        font-size: 16px !important;
        font-weight: bold !important;
    }
    
    /* Multiselect */
    .stMultiSelect > div > div {
        background-color: #2a2a2a;
        color: #ffffff;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background-color: #0066cc;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #0066cc;
    }
    
    /* Success messages - texto blanco */
    .stSuccess {
        background-color: #1a4d1a;
        border: 1px solid #2d5a2d;
        color: #ffffff !important;
    }
    
    /* Warning messages - texto blanco */
    .stWarning {
        background-color: #4d3d1a;
        border: 1px solid #5a4d2d;
        color: #ffffff !important;
    }
    
    /* Error messages - texto blanco */
    .stError {
        background-color: #4d1a1a;
        border: 1px solid #5a2d2d;
        color: #ffffff !important;
    }
    
    /* Info messages - texto blanco */
    .stInfo {
        background-color: #1a3d4d;
        border: 1px solid #2d4d5a;
        color: #ffffff !important;
    }
    
    /* Forzar texto blanco en todos los elementos */
    .stMarkdown p, .stMarkdown div, .stText {
        color: #ffffff !important;
    }
    
    /* Forzar texto blanco en alertas y notificaciones */
    .stAlert > div, .stSuccess > div, .stWarning > div, .stError > div, .stInfo > div {
        color: #ffffff !important;
    }
    
    /* Forzar texto blanco en métricas */
    .metric-container, [data-testid="metric-container"] {
        color: #ffffff !important;
    }
    
    /* Forzar texto blanco en elementos específicos de texto y widgets */
    .stMarkdown p, .stMarkdown div, .stMarkdown span {
        color: #ffffff !important;
    }
    
    /* Labels de widgets específicos */
    .stSelectbox > label, .stMultiSelect > label, .stTextInput > label, .stNumberInput > label {
        color: #ffffff !important;
    }
    
    /* Sidebar específico */
    .stSidebar .stMarkdown p, .stSidebar .stMarkdown div, .stSidebar .stMarkdown span {
        color: #ffffff !important;
    }
    
    /* Métricas específicas */
    [data-testid="metric-container"] {
        color: #ffffff !important;
    }
    
    [data-testid="metric-container"] > div {
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)


def get_color_scheme():
    """
    Retorna el esquema de colores para gráficos.
    
    Returns:
        dict: Diccionario con colores del tema
    """
    return {
        'background': '#000000',
        'text': '#ffffff',
        'primary': '#ffffff',      # Cambiado a blanco
        'secondary': '#ffffff',    # Cambiado a blanco  
        'accent': '#ffffff',       # Cambiado a blanco
        'warning': '#ffffff',      # Cambiado a blanco
        'error': '#ffffff',        # Cambiado a blanco
        'success': '#ffffff',      # Cambiado a blanco
        'info': '#ffffff'          # Cambiado a blanco
    }

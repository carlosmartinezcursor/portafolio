# 📊 Portafolio de Inversión con Optimización de Markowitz

## 🎯 Descripción del Proyecto

Este proyecto implementa una aplicación interactiva para analizar y optimizar un portafolio de inversión utilizando el modelo de Markowitz. La aplicación está desarrollada con **Python puro** (NumPy/SciPy) sin dependencias externas como PyPortfolioOpt, asegurando que los estudiantes comprendan la lógica subyacente.

## 🔧 Características Técnicas

- **Backend**: Python puro con NumPy, SciPy, Pandas
- **Frontend**: Streamlit con tema oscuro profesional
- **Visualizaciones**: Plotly para gráficos interactivos
- **Datos**: 20 empresas + S&P 500 (Octubre 2020 - Septiembre 2025)

## 📋 Funcionalidades

### ✅ Funcionalidad 0: Carga y Preparación de Datos
- Carga automática del archivo CSV
- Validación de datos faltantes
- Exclusión automática de activos incompletos
- Mensajes informativos al usuario

### 🚧 Funcionalidad 1: Inputs y Configuración Inicial
- Selección de activos del universo de inversión
- Configuración de pesos forzados
- Input de tasa libre de riesgo
- Validación de parámetros

### 🚧 Funcionalidad 2: Análisis de Riesgos y Dependencias
- Matriz de correlaciones con mapa de calor
- Comparación de escenarios (pesos iguales vs optimizado)
- Métricas de riesgo y rendimiento

### 🚧 Funcionalidad 3: Optimización y Frontera Eficiente
- Cálculo de frontera eficiente con SciPy
- Visualización de portafolios óptimos
- Línea del mercado de capitales (CML)

### 🚧 Funcionalidad 4: Resultados Finales y Validación Histórica
- Tabla de pesos del portafolio optimizado
- Gráfico de evolución histórica
- Comparación con benchmark (S&P 500)

## 🚀 Instalación y Uso

### 1. Configurar Entorno Virtual
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 2. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar la Aplicación
```bash
streamlit run main.py
```

La aplicación se abrirá en `http://localhost:8501`

## 📁 Estructura del Proyecto

```
portafolio/
├── datos/
│   ├── portafolio_21_activos.csv
│   └── documentodefuncionalidades.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Funcionalidad 0
│   └── theme.py               # Tema oscuro
├── main.py                    # Aplicación principal
├── requirements.txt
└── README.md
```

## 🎨 Tema Visual

La aplicación utiliza un **tema oscuro profesional** inspirado en terminales financieras como Bloomberg:
- Fondo negro con colores de contraste
- Azul, verde, rojo, morado para elementos interactivos
- Navegación por menú lateral
- Estilo elegante y profesional

## 📊 Datos del Portafolio

- **Período**: Octubre 2020 - Septiembre 2025
- **Frecuencia**: Datos mensuales
- **Activos**: 20 empresas + S&P 500
- **Formato**: Precios de cierre ajustados

### Activos Incluidos
```
AAPL, AMZN, BAC, CVX, DIS, GOOGL, IBM, JNJ, JPM, KO, 
MA, META, MSFT, NFLX, NVDA, PFE, TSLA, V, WMT, XOM, ^GSPC
```

## 🔬 Metodología

### Optimización de Markowitz
- Implementación desde cero con SciPy
- Optimización cuadrática para frontera eficiente
- Cálculo de ratio de Sharpe
- Línea del mercado de capitales

### Análisis de Riesgo
- Matriz de covarianzas
- Correlaciones entre activos
- Volatilidad y rendimiento esperado
- Diversificación del portafolio

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**
- **Streamlit**: Interfaz web interactiva
- **Pandas**: Manipulación de datos
- **NumPy**: Cálculos numéricos
- **SciPy**: Optimización cuadrática
- **Plotly**: Visualizaciones interactivas

## 📝 Notas de Desarrollo

- **Sin PyPortfolioOpt**: Todos los cálculos implementados desde cero
- **Validación robusta**: Manejo de datos faltantes y errores
- **Tema consistente**: Diseño oscuro en toda la aplicación
- **Navegación intuitiva**: Menú lateral para cada funcionalidad

## 🎓 Objetivo Educativo

Este proyecto está diseñado para que los estudiantes:
1. Comprendan la teoría detrás de la optimización de portafolios
2. Implementen los cálculos sin bibliotecas externas
3. Visualicen los conceptos de riesgo y rendimiento
4. Analicen la diversificación y correlaciones

## 📞 Soporte

Para consultas o problemas, revisa la documentación en `datos/documentodefuncionalidades.md` o contacta al equipo de desarrollo.

# 📘 Manual Técnico - Portfolio Optimization Engine

**Guía completa de instalación, ejecución y personalización**

---

## 📋 Índice

1. [Requisitos del Sistema](#requisitos-del-sistema)
2. [Instalación](#instalación)
3. [Ejecución del Motor](#ejecución-del-motor)
4. [Personalización Avanzada](#personalización-avanzada)
5. [Estructura del Proyecto](#estructura-del-proyecto)
6. [Interpretación de Resultados](#interpretación-de-resultados)
7. [Solución de Problemas](#solución-de-problemas)
8. [Preguntas Frecuentes (FAQ)](#preguntas-frecuentes-faq)

---

## 📦 Requisitos del Sistema

### Verificar Python
```bash
python --version
```

**Requisito:** Python **3.8 o superior**

Si no tienes Python instalado:
- **Windows:** Descargar desde [python.org](https://www.python.org/downloads/)
- **macOS:** `brew install python`
- **Linux:** `sudo apt-get install python3 python3-pip`

---

## 🚀 Instalación

### 1. Descargar el Proyecto

```bash
# Opción 1: Clonar repositorio (si está en GitHub)
git clone https://github.com/tu-usuario/Modelo-de-optimizacion-de-portafolios-.git
cd Modelo-de-optimizacion-de-portafolios-

# Opción 2: Descomprimir archivo ZIP descargado desde GitHub
# (El nombre de la carpeta dependerá de cómo lo descargues/extraigas)
cd portfolio-optimizer  # Ajusta según tu nombre de carpeta local
```

### 2. Instalar Dependencias

```bash
python -m pip install -r requirements.txt
```

**Dependencias instaladas:**
- `yfinance>=0.2.28` - Descarga de datos desde Yahoo Finance
- `pandas>=2.0.0` - Manipulación de datos
- `numpy>=1.24.0` - Cálculos numéricos
- `scipy>=1.10.0` - Optimización y estadística
- `matplotlib>=3.7.0` - Gráficos
- `seaborn>=0.12.0` - Visualizaciones avanzadas

**Nota para Windows:** Si encuentras errores de compilación, las dependencias se instalarán automáticamente con wheels pre-compilados.

### 3. Verificar Instalación

```bash
python -c "import yfinance, pandas, numpy, scipy, matplotlib, seaborn; print('✅ Todas las dependencias instaladas correctamente')"
```

---

## 💻 Ejecución del Motor

### Opción 1: Script Completo (Recomendado)

```bash
cd src
python portfolio_engine.py
```

**Flujo de ejecución (6 módulos secuenciales):**

```
MÓDULO 1/6: DESCARGA DE DATOS
  ↓ Descarga datos históricos de Yahoo Finance
  ↓ Período: Configurable (default: 2 años para optimización + 1 año para backtest)
  ↓ Activos: 7 por default (GGAL.BA, YPFD.BA, ALUA.BA, GOOGL, BTC-USD, MSFT, KO)

MÓDULO 2/6: ESTADÍSTICAS DESCRIPTIVAS
  ↓ Calcula retornos diarios → anualizados (×252)
  ↓ Volatilidad diaria → anualizada (×√252)
  ↓ Sharpe Ratio individual: (R - Rf) / σ
  ↓ Matriz de covarianza anualizada
  ↓ Matriz de correlación

MÓDULO 3/6: OPTIMIZACIÓN Y FRONTERA EFICIENTE
  ↓ Optimización Libre (sin restricciones)
  ↓ Optimización Gestionada (con restricciones por tipo de activo)
  ↓ Comparación: Sharpe Libre vs Gestionada
  ↓ Genera: efficient_frontier.png + reporte_portfolio.md

MÓDULO 4/6: ANÁLISIS DE RIESGO (VaR/CVaR t-Student) [EX-ANTE]
  ↓ 10,000 simulaciones Monte Carlo por escenario
  ↓ 3 escenarios: Conservador (df=3), Esperado (df≈5), Normal
  ↓ Proyección: "Espero perder máximo X% con 95% confianza"
  ↓ Métricas: VaR y CVaR (diarios y anualizados)

MÓDULO 5/6: BACKTESTING (VALIDACIÓN HISTÓRICA) [EX-POST]
  ↓ Validación out-of-sample (datos posteriores al período de optimización)
  ↓ Simula Active (rebalanceo mensual) vs Passive (buy-hold)
  ↓ Comisiones: 0.5% por operación
  ↓ Análisis de fricción: ¿Vale la pena el rebalanceo?
  ↓ Comparación vs Benchmark (SPY)
  ↓ Genera: backtest_results.png + reporte_backtesting.md

MÓDULO 6/6: STRESS TESTING (ESCENARIOS EXTREMOS)
  ↓ Escenario 1: Crash Global (S&P -20%, Tech -25%, BTC -40%)
  ↓ Escenario 2: Crisis Argentina (stocks locales -40%)
  ↓ Escenario 3: Recuperación Agresiva (+20% todos)
  ↓ Genera: stress_test.png + reporte_stress_test.md
```

**Tiempo total:** ~30-45 segundos (depende de conexión a internet)

**Resultado:** 3 gráficos PNG + 3 reportes Markdown en `outputs/`

### Opción 2: Jupyter Notebook (Interactivo)

```bash
cd notebooks
jupyter notebook analisis_portfolio.ipynb
```

**Ventajas:**
- Ejecutar módulos paso a paso
- Modificar parámetros en tiempo real
- Experimentar con diferentes configuraciones
- Visualizar resultados intermedios

---

## 🔧 Personalización Avanzada

### 1. Cambiar Activos Analizados

**Edita:** `src/portfolio_engine.py`, función `main()` (línea ~2925)

```python
tickers = [
    'GGAL.BA',      # Banco Galicia (Argentina)
    'YPFD.BA',      # YPF (Argentina)
    'BBAR.BA',      # BBVA Argentina (Argentina)
    'AAPL',         # Apple (Global)
    'MSFT',         # Microsoft (Global)
    'JPM',          # JPMorgan Chase (Global)
    'BTC-USD'       # Bitcoin (Cripto)
]
```

**Activos disponibles:**

🇦🇷 **Argentinos (.BA):**
- Financiero: GGAL.BA, BBAR.BA, BMA, SUPV.BA
- Energía: YPFD.BA, PAMP.BA, TGS.BA, CEPU.BA
- Industrial: ALUA.BA, TXAR.BA, LOMA.BA

🌎 **Globales:**
- Tech: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA
- Financiero: JPM, BAC, GS, WFC, C, MS
- Consumo: KO, PEP, JNJ, PG, DIS, NKE
- ETFs: SPY, QQQ, IWM, VTI

₿ **Criptomonedas (-USD):**
- BTC-USD, ETH-USD, ADA-USD, SOL-USD, XRP-USD

**⚠️ Nota:** Los bonos argentinos (AL30, GD30, AE38) **NO están disponibles** en Yahoo Finance. Usa ETFs de bonos como TLT, AGG, EMB.

### 2. Cambiar Período de Análisis

**Edita:** `src/portfolio_engine.py`, función `main()` (línea ~2939)

**Opción 1: Cambiar años (Recomendado)**

```python
# Configuración de períodos
ANALYSIS_YEARS = 2  # ← Años de datos para optimización (mínimo: 2, óptimo: 3-5)
BACKTEST_YEARS = 1  # ← Años para backtesting (debe ser < ANALYSIS_YEARS)
```

**Ejemplo: Análisis de 10 años con backtesting de 2 años**
```python
ANALYSIS_YEARS = 10  # 10 años para optimizar
BACKTEST_YEARS = 2   # 2 años para validar
```

**Cómo funciona el Train/Test Split:**
```
Hoy: 2026-02-17

ANALYSIS_YEARS = 2, BACKTEST_YEARS = 1
↓
Train Set: 2023-02-18 a 2025-02-17 (2 años) → Optimización
Test Set:  2025-02-17 a 2026-02-17 (1 año)  → Backtesting
```

**Opción 2: Fechas específicas (Avanzado)**

```python
# Comentar las líneas de ANALYSIS_YEARS/BACKTEST_YEARS y agregar:
start_date_optimization = '2020-01-01'
end_date_optimization = '2025-01-01'
```

**Recomendaciones:**
- **Mínimo:** 2 años (para capturar volatilidad)
- **Óptimo:** 3-5 años (balance entre estabilidad y relevancia)
- **Máximo:** 10 años (datos muy antiguos pueden ser irrelevantes)

**⚠️ Importante:** `BACKTEST_YEARS` siempre debe ser **menor** que `ANALYSIS_YEARS`.

### 3. Ajustar Tasa Libre de Riesgo

**Edita:** `src/portfolio_engine.py`, función `main()` (línea ~2965)

```python
# Inicializar optimizador con tasa libre de riesgo personalizada
optimizer = PortfolioOptimizer(
    tickers, 
    start_date=start_date_optimization,
    end_date=end_date_optimization,
    risk_free_rate=0.05  # ← Cambiar aquí (default: 5%)
)
```

**Ejemplos:**
```python
risk_free_rate=0.03  # 3% (más conservador, baja Sharpe)
risk_free_rate=0.07  # 7% (comparar con bonos de alto rendimiento)
risk_free_rate=0.00  # 0% (solo para comparación académica, incorrecto)
```

**Impacto:** Afecta el cálculo del Sharpe Ratio en **TODOS** los módulos:
- Optimización (Max Sharpe)
- Estadísticas descriptivas
- Backtesting (Active, Passive, Benchmark)

**Fórmula:** `Sharpe = (Retorno - Risk_Free_Rate) / Volatilidad`

### 4. Configurar Restricciones de Peso (Bounds)

**Edita:** `src/portfolio_engine.py`, función `main()` (línea ~3007)

**Sistema actual (Dinámico):**
```python
# El código genera automáticamente:
bounds_managed = {
    'GGAL.BA': (0.00, 0.20),   # Argentinos: Max 20%
    'YPFD.BA': (0.00, 0.20),
    'ALUA.BA': (0.00, 0.20),
    'GOOGL': (0.15, 0.35),     # Líder global: 15-35%
    'BTC-USD': (0.00, 0.10),   # Cripto: Max 10%
    'MSFT': (0.10, 0.30),      # Otros globales: 10-30%
    'KO': (0.10, 0.30),
}
```

**Personalización manual (Avanzado):**

Reemplaza el bloque `bounds_managed = {}` con:

```python
# Portfolio conservador
bounds_managed = {
    'GGAL.BA': (0.0, 0.10),   # Max 10% en cada argentino
    'YPFD.BA': (0.0, 0.10),
    'ALUA.BA': (0.0, 0.10),
    'BTC-USD': (0.0, 0.05),   # Max 5% en Bitcoin
    'GOOGL': (0.15, 0.40),    # Min 15%, Max 40% en Google
    'MSFT': (0.15, 0.40),     # Min 15%, Max 40% en Microsoft
    'KO': (0.10, 0.30),       # Entre 10% y 30% en Coca-Cola
}

# Portfolio agresivo
bounds_managed = {
    'GGAL.BA': (0.0, 0.30),   # Hasta 30% en argentinos
    'YPFD.BA': (0.0, 0.30),
    'ALUA.BA': (0.0, 0.30),
    'BTC-USD': (0.0, 0.20),   # Hasta 20% en Bitcoin
    'GOOGL': (0.05, 0.30),    # Menor obligatoriedad
    'MSFT': (0.05, 0.30),
    'KO': (0.00, 0.20),       # Opcional
}
```

**Regla:** La suma de los límites superiores debe ser ≥100% para que la optimización sea factible.

### 5. Ajustar Parámetros de Backtesting

**Edita:** `src/portfolio_engine.py`, línea ~3075 (llamada a `run_backtest`)

```python
backtest_results = optimizer.run_backtest(
    max_sharpe_weights_free,           # Pesos a validar
    initial_capital=1000000,           # ← Capital inicial en USD (default: 1M)
    lookback_period='1y',              # ← Período: '6mo', '1y', '2y', '180d'
    benchmark_ticker='SPY',            # ← Benchmark: 'SPY', 'QQQ', '^MERV'
    rebalance=True,                    # ← True=Mensual, False=Buy-Hold
    commission_pct=0.005               # ← Comisión: 0.005=0.5%, 0.001=0.1%
)
```

**Ejemplos:**

```python
# Backtest con capital más pequeño y comisiones bajas (broker barato)
backtest_results = optimizer.run_backtest(
    max_sharpe_weights_free,
    initial_capital=100000,      # USD 100,000
    lookback_period='2y',        # 2 años
    benchmark_ticker='QQQ',      # Nasdaq 100
    rebalance=True,
    commission_pct=0.001         # 0.1% por operación
)

# Backtest pasivo sin rebalanceo
backtest_results = optimizer.run_backtest(
    max_sharpe_weights_free,
    initial_capital=1000000,
    lookback_period='1y',
    benchmark_ticker='SPY',
    rebalance=False,             # Sin rebalanceo
    commission_pct=0.005
)
```

**Nota:** El módulo 5 siempre ejecuta **ambas** estrategias (Active y Passive) para comparar.

### 6. Cambiar Nivel de Confianza del VaR

**Edita:** `src/portfolio_engine.py`, línea ~3053 (llamada a `calculate_var_cvar`)

```python
# VaR al 95% (default)
risk_metrics = optimizer.calculate_var_cvar(
    max_sharpe_weights_free, 
    confidence_level=0.95,       # ← 95% confianza (default)
    n_simulations=10000,
    use_students_t=True,
    df_conservative=3
)

# VaR al 99% (más conservador)
risk_metrics = optimizer.calculate_var_cvar(
    max_sharpe_weights_free, 
    confidence_level=0.99,       # ← 99% confianza
    n_simulations=10000,
    use_students_t=True,
    df_conservative=3
)
```

**Interpretación:**
- **95%:** "En el 5% peor de los días, perderé más de X%"
- **99%:** "En el 1% peor de los días, perderé más de X%" (más estricto)

**Recomendación:** Mantener 95% (estándar de la industria, Basel III).

### 7. Personalizar Escenarios de Stress Testing

**Edita:** `src/portfolio_engine.py`, función `run_stress_test()` (línea ~1750)

**Localiza el diccionario `scenarios`:**

```python
scenarios = {
    'Crash Global': {
        'GGAL.BA': -0.20,
        'YPFD.BA': -0.20,
        'ALUA.BA': -0.20,
        'GOOGL': -0.25,
        'MSFT': -0.25,
        'KO': -0.15,
        'BTC-USD': -0.40,
    },
    'Crisis Argentina': {
        'GGAL.BA': -0.40,
        'YPFD.BA': -0.40,
        'ALUA.BA': -0.40,
        'GOOGL': -0.05,
        'MSFT': -0.05,
        'KO': -0.05,
        'BTC-USD': -0.10,
    },
    'Recuperación': {
        'GGAL.BA': 0.20,
        'YPFD.BA': 0.20,
        'ALUA.BA': 0.20,
        'GOOGL': 0.20,
        'MSFT': 0.20,
        'KO': 0.20,
        'BTC-USD': 0.20,
    },
}
```

**Agregar un escenario personalizado:**

```python
scenarios = {
    # ... escenarios existentes ...
    
    'Hiperinflación Argentina': {
        'GGAL.BA': -0.50,   # -50%
        'YPFD.BA': -0.50,
        'ALUA.BA': -0.50,
        'GOOGL': 0.00,      # Sin impacto
        'MSFT': 0.00,
        'KO': 0.00,
        'BTC-USD': 0.30,    # +30% (safe haven)
    },
    
    'Bull Market Tecnológico': {
        'GGAL.BA': 0.10,
        'YPFD.BA': 0.10,
        'ALUA.BA': 0.10,
        'GOOGL': 0.40,      # +40%
        'MSFT': 0.40,
        'KO': 0.15,
        'BTC-USD': 0.50,    # +50%
    },
}
```

**⚠️ Importante:** Los tickers en el diccionario deben coincidir **exactamente** con los de `tickers` en `main()`.

### 8. Cambiar Capital Inicial

**Edita:** `src/portfolio_engine.py`, función `main()` (línea ~2955)

```python
INITIAL_CAPITAL = 1000000  # ← Cambiar aquí (default: USD 1,000,000)
```

**Ejemplos:**
```python
INITIAL_CAPITAL = 100000    # USD 100,000
INITIAL_CAPITAL = 500000    # USD 500,000
INITIAL_CAPITAL = 10000000  # USD 10,000,000
```

**Impacto:** Afecta los reportes de Backtesting y Stress Testing (capital final, comisiones en USD).

---

## 📁 Estructura del Proyecto

```
Modelo-de-optimizacion-de-portafolios-/
│
├── data/                      # Datos descargados (generado automáticamente)
│   └── *.csv                  # Archivos CSV de precios históricos
│
├── notebooks/                 # Jupyter notebooks para análisis interactivo
│   └── analisis_portfolio.ipynb
│
├── src/                       # Código fuente
│   └── portfolio_engine.py    # Motor principal (3,360 líneas)
│
├── outputs/                   # Resultados generados (se sobrescriben en cada ejecución)
│   ├── efficient_frontier.png       # Frontera eficiente + matrices (300 DPI)
│   ├── backtest_results.png         # Equity curve + drawdown (300 DPI)
│   ├── stress_test.png              # Escenarios extremos (300 DPI)
│   ├── reporte_portfolio.md         # Análisis Ex-Ante (~350 líneas)
│   ├── reporte_backtesting.md       # Validación Ex-Post (~290 líneas)
│   └── reporte_stress_test.md       # Análisis de Escenarios (~300 líneas)
│
├── requirements.txt           # Dependencias Python
├── README.md                  # Documentación principal (metodología, resultados)
└── INSTRUCCIONES.md           # Este archivo (manual técnico)
```

**⚠️ Nota:** Los archivos en `outputs/` se **sobrescriben** en cada ejecución. Si quieres guardar versiones anteriores, renombra los archivos manualmente antes de volver a ejecutar.

---

## 📊 Interpretación de Resultados

### 1. Terminal Output

**Módulo 1 - Descarga de Datos:**
```
[i] Activos descargados exitosamente:
    GGAL.BA: 1258 días
    YPFD.BA: 1258 días
    ...
```
✅ Verifica que todos los activos se descargaron correctamente.

**Módulo 2 - Estadísticas:**
```
[i] Estadisticas descriptivas individuales:

Ticker       Ret.Anual  Vol.Anual  Sharpe   
GGAL.BA      12.34%     45.67%     0.16
YPFD.BA      8.92%      38.45%     0.10
...
```
- **Ret.Anual:** Retorno esperado (media histórica × 252)
- **Vol.Anual:** Volatilidad (std × √252)
- **Sharpe:** (Retorno - 5%) / Volatilidad

**Módulo 3 - Optimización:**
```
[✓] CARTERA DE MINIMA VOLATILIDAD:
    Retorno Esperado: 8.45%
    Volatilidad:      18.23%
    Sharpe Ratio:     0.19

[✓] CARTERA DE MAXIMO SHARPE (LIBRE):
    Retorno Esperado: 15.67%
    Volatilidad:      25.34%
    Sharpe Ratio:     0.42
    
    Pesos:
    GOOGL:   35.0%
    MSFT:    30.0%
    ...
```

**Módulo 4 - VaR/CVaR:**
```
[i] VaR/CVaR con distribucion t-Student (95% confianza):

Escenario        df    VaR (1d)   CVaR (1d)  VaR (anual)  CVaR (anual)
Conservador      3     -4.93%     -8.51%     -78%         -135%
Esperado         5     -3.21%     -5.34%     -51%         -85%
Normal (Gauss)   ∞     -2.89%     -3.65%     -46%         -58%
```
**Interpretación:**
- **VaR (1d):** Pérdida máxima en 1 día con 95% confianza
- **VaR (anual):** Pérdida máxima en 1 año con 95% confianza (extrapolación, menos confiable)

**Módulo 5 - Backtesting:**
```
[i] Metricas finales:

Estrategia       Ret.Total  Sharpe  Max DD   Vol.Anual
Active (Rebal.)  +24.16%    0.84    -18.23%  28.76%
Passive (B&H)    +14.92%    0.52    -22.67%  28.64%
Benchmark (SPY)  +18.45%    0.68    -15.89%  27.12%

[i] Analisis de Friccion:
    Comisiones Extra (Active): $5,844.46
    Diferencia de Retorno:     +9.24%
    Rebalanceo valio la pena:  SÍ ✅
```

**Módulo 6 - Stress Testing:**
```
[i] Impacto por escenario:

Escenario            Capital Final  Impacto
Crash Global         $762,000       -23.8%
Crisis Argentina     $746,000       -25.4%
Recuperacion         $1,200,000     +20.0%
```

### 2. Gráficos PNG

**efficient_frontier.png (4 subplots):**
- **Top-Left:** Frontera eficiente con puntos óptimos marcados (estrella verde = Min Vol, estrella roja = Max Sharpe)
- **Top-Right:** Matriz de correlación (colores cálidos = alta correlación, fríos = baja)
- **Bottom-Left:** Composición de carteras (barras apiladas por activo)
- **Bottom-Right:** Matriz de covarianza (riesgo conjunto entre pares de activos)

**backtest_results.png (4 subplots):**
- **Top-Left:** Equity Curve (Active azul, Passive naranja, Benchmark morado)
- **Top-Right:** Drawdown desde máximos históricos (áreas de gradiente)
- **Bottom-Left:** Distribución de retornos diarios (histogramas superpuestos)
- **Bottom-Right:** Comparación de métricas (barras: Sharpe, DD, Retorno)

**stress_test.png (2 subplots):**
- **Left:** Capital final por escenario (barras comparativas)
- **Right:** Impacto en USD (pérdida/ganancia absoluta)

### 3. Reportes Markdown

**reporte_portfolio.md:**
- Secciones: Estadísticas, Covarianza, Correlación, Carteras Optimizadas, VaR/CVaR, Comparación Libre vs Gestionada

**reporte_backtesting.md:**
- Secciones: Metodología Out-of-Sample, Performance Active/Passive, Análisis de Fricción, Validación de VaR

**reporte_stress_test.md:**
- Secciones: Escenario 1, Escenario 2, Escenario 3, Análisis de Resiliencia

---

## 🆘 Solución de Problemas

### Error: "No se pudieron descargar datos para [ticker]"

**Causa:** El ticker no existe en Yahoo Finance o está descontinuado.

**Solución:**
1. Verifica el ticker en [finance.yahoo.com](https://finance.yahoo.com/)
2. Para activos argentinos, prueba con/sin `.BA` (ej: `GGAL` vs `GGAL.BA`)
3. Reemplaza el ticker por uno similar del mismo sector

**Ejemplo:**
```python
# Si falla:
tickers = ['AL30.BA', 'GGAL.BA', ...]

# Prueba:
tickers = ['GGAL.BA', 'YPFD.BA', ...]  # Sin AL30.BA (no disponible)
```

### Error: "ModuleNotFoundError: No module named 'yfinance'"

**Causa:** Dependencias no instaladas.

**Solución:**
```bash
python -m pip install -r requirements.txt --upgrade
```

### Error: "UnicodeEncodeError" en Windows

**Causa:** La terminal de Windows no soporta UTF-8 por default.

**Solución:**
```bash
# Ejecutar con flag unicode
python -u portfolio_engine.py

# O cambiar encoding de la terminal
chcp 65001
python portfolio_engine.py
```

### Error: "Optimization failed: Maximum iterations exceeded"

**Causa:** El optimizador no convergió (restricciones muy estrictas o datos problemáticos).

**Solución:**
- El código automáticamente usa **pesos equiponderados** como fallback
- Verifica que los límites superiores de `bounds_dict` sumen ≥100%
- Revisa que los datos descargados sean suficientes (mínimo 2 años)

**Ejemplo de restricciones inválidas:**
```python
# INCORRECTO (suma max = 60%)
bounds_managed = {
    'GGAL.BA': (0.0, 0.10),
    'YPFD.BA': (0.0, 0.10),
    'ALUA.BA': (0.0, 0.10),
    'GOOGL': (0.0, 0.10),
    'MSFT': (0.0, 0.10),
    'KO': (0.0, 0.10),
}

# CORRECTO (suma max = 180%)
bounds_managed = {
    'GGAL.BA': (0.0, 0.20),
    'YPFD.BA': (0.0, 0.20),
    'ALUA.BA': (0.0, 0.20),
    'GOOGL': (0.0, 0.40),
    'MSFT': (0.0, 0.40),
    'KO': (0.0, 0.40),
}
```

### Los gráficos no se generan

**Causa:** Error en matplotlib o seaborn.

**Solución:**
```bash
# Reinstalar dependencias de visualización
python -m pip install matplotlib seaborn --upgrade --force-reinstall
```

### Error: "KeyError: '[ticker]' not found in axis"

**Causa:** Los tickers en `scenarios` (stress testing) no coinciden con los de `tickers`.

**Solución:**
- Edita `run_stress_test()` (línea ~1750) y asegúrate que los tickers en `scenarios` sean exactamente los mismos que en `main()`.

**Ejemplo:**
```python
# En main():
tickers = ['GGAL.BA', 'AAPL', 'BTC-USD']

# En scenarios (debe coincidir):
scenarios = {
    'Crash Global': {
        'GGAL.BA': -0.20,   # ✅ Correcto
        'AAPL': -0.25,      # ✅ Correcto
        'BTC-USD': -0.40,   # ✅ Correcto
    }
}
```

### La ejecución es muy lenta (>5 minutos)

**Causa:** Conexión a internet lenta o muchos activos.

**Solución:**
- Reduce el número de activos (máximo 10-12)
- Reduce `n_simulations` en `calculate_var_cvar` (de 10,000 a 5,000)
- Verifica tu conexión a internet

---

## ❓ Preguntas Frecuentes (FAQ)

### 1. ¿Los reportes se sobrescriben cada vez que ejecuto el código?

**R:** **SÍ**. Todos los archivos en `outputs/` se regeneran en cada ejecución con los últimos datos y configuración. Si quieres guardar versiones, renombra los archivos manualmente antes de volver a ejecutar.

**Ejemplo:**
```bash
# Antes de ejecutar nuevamente
cd outputs
mv reporte_portfolio.md reporte_portfolio_2026-02-17.md
mv backtest_results.png backtest_results_2026-02-17.png
```

### 2. ¿Cómo cambio el período de análisis a 3 años?

**R:** Ver sección **"Personalización Avanzada → 2. Cambiar Período de Análisis"**.

**Forma rápida:**
```python
# En main() (línea ~2939)
ANALYSIS_YEARS = 3  # ← Cambiar aquí
BACKTEST_YEARS = 1
```

### 3. ¿Puedo usar este código para trading en vivo?

**R:** ⚠️ **NO recomendado** sin ajustes significativos. Este es un sistema de **asignación estratégica** (largo plazo), no trading táctico. Necesitarías agregar:
- Integración con broker (API de IOL, BullMarket, Interactive Brokers)
- Órdenes límite/stop
- Manejo de liquidez (volúmenes, spreads)
- Ejecución gradual (para órdenes grandes)
- Monitoreo en tiempo real

### 4. ¿Qué significa "df=3" en el VaR conservador?

**R:** Son los **grados de libertad** de la distribución t-Student:
- **df=3:** Muy conservador (colas muy pesadas), recomendado para asignación de capital
- **df≈5:** Más realista, estimado por Maximum Likelihood Estimation (MLE) de datos históricos
- **df=∞:** Normal (no recomendado, subestima riesgo)

**Menor df = Más peso en eventos extremos**

### 5. ¿Por qué el rebalanceo activo tuvo menor retorno que el pasivo?

**R:** Puede ocurrir si el mercado tendió en una dirección sin reversión. El rebalanceo vale la pena en mercados **mean-reverting** (reversión a la media), no en tendencias fuertes. Revisa el **análisis de fricción** en `reporte_backtesting.md`.

**Criterio:**
```
Rebalanceo vale la pena si:
Retorno_Active - Retorno_Passive > (Comisiones_Extra / Capital_Inicial)
```

### 6. ¿Cómo interpreto el VaR anualizado de -78%?

**R:** Es el **peor caso anual** con 95% confianza bajo distribución t-Student conservadora (df=3). NO significa que perderás 78% cada año, sino que en el **5% peor de los años**, la pérdida podría alcanzar ese nivel.

**Importante:** La extrapolación `VaR_daily × √252` tiene limitaciones. El VaR diario es más confiable.

### 7. ¿Cuándo debo usar la optimización "Gestionada" en lugar de "Libre"?

**R:** Usa **Gestionada** cuando:
- Tienes políticas de inversión (ej: "max 20% en un activo")
- Quieres limitar exposición a activos volátiles (cripto, argentinos)
- Necesitas cumplir requisitos regulatorios

El **Sharpe Gestionado** será ligeramente menor, pero tendrás mayor control de riesgo.

### 8. ¿Por qué mi Sharpe Ratio es diferente a otros cálculos?

**R:** Este código usa **tasa libre de riesgo (rf=5%)** correctamente:
```
Sharpe = (Retorno - Rf) / Volatilidad
```

Muchos calculadores online usan `Sharpe = Retorno / Volatilidad` (rf=0%), lo cual es **incorrecto**.

### 9. ¿Qué hacer si Yahoo Finance no tiene datos para un ticker argentino?

**R:** Algunos tickers están descontinuados o tienen nombre diferente:
- Prueba con/sin `.BA` (ej: `GGAL` vs `GGAL.BA`)
- Verifica en [finance.yahoo.com](https://finance.yahoo.com/) si el ticker existe
- Usa otro activo similar del mismo sector
- Los **bonos argentinos (AL30, GD30, AE38) NO están disponibles** → usa ETFs de bonos (TLT, AGG, EMB)

### 10. ¿Puedo agregar más de 10 activos?

**R:** Sí, pero ten en cuenta:
- **Más activos = Mayor tiempo de ejecución** (~5 seg por activo)
- **Optimización más compleja** (puede no converger con >15 activos)
- **Beneficio de diversificación se satura** después de 8-12 activos

**Recomendación:** Mantener entre 5-12 activos bien seleccionados.

---

## 📚 Referencias Técnicas

### Documentación de Librerías

- **yfinance:** [GitHub](https://github.com/ranaroussi/yfinance) | [PyPI](https://pypi.org/project/yfinance/)
- **scipy.optimize:** [Docs](https://docs.scipy.org/doc/scipy/reference/optimize.html)
- **pandas:** [Docs](https://pandas.pydata.org/docs/)
- **matplotlib:** [Docs](https://matplotlib.org/stable/contents.html)

### Algoritmos Implementados

**Optimización:**
- `scipy.optimize.minimize` con método `SLSQP` (Sequential Least Squares Programming)
- Restricciones: Pesos suman 100%, límites por activo (bounds)

**Estadística:**
- `scipy.stats.t` (distribución t-Student)
- Maximum Likelihood Estimation (MLE) para estimar df

**Anualización:**
- Retornos: `CAGR = (1 + ret_total)^(252/n_days) - 1`
- Volatilidad: `Vol_annual = Vol_daily × √252`

---

## 📧 Soporte

Para preguntas técnicas o mejoras al código:
- **Documentación oficial:** [scipy.org](https://scipy.org), [pandas.pydata.org](https://pandas.pydata.org)
- **Yahoo Finance API:** [github.com/ranaroussi/yfinance](https://github.com/ranaroussi/yfinance)
- **Teoría de Carteras:** Markowitz (1952), Sharpe (1966)

---

**Desarrollado por Jorge Iván Juárez A. - Lic. en Economía especializado en mercado de capitales**

---

*Última actualización: Febrero 2026*


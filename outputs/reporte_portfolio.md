# Reporte de Optimización de Cartera
## Análisis Cuantitativo del Mercado Argentino y Global

**Fecha de Generación:** 2026-02-17 19:58:57  
**Período Analizado (Train Set):** 2023-02-18 a 2025-02-17  
**Analista:** Jorge Iván Juárez A. - Lic. en Economía

---

> **📊 Nota Metodológica:** Este reporte contiene el análisis **ex-ante** basado en datos históricos 
> del período indicado (train set). Los pesos óptimos calculados aquí se validan posteriormente 
> en el **Reporte de Backtesting** usando datos **out-of-sample** (test set) para asegurar 
> robustez y evitar overfitting.

---

## 1. Activos Bajo Análisis

Los activos seleccionados para el análisis son:

- **GGAL.BA** - [ARG] Equity Argentina
- **YPFD.BA** - [ARG] Equity Argentina
- **ALUA.BA** - [ARG] Equity Argentina
- **GOOGL** - [GLOBAL] Equity Global
- **BTC-USD** - [CRYPTO] Criptomoneda
- **MSFT** - [GLOBAL] Equity Global
- **KO** - [GLOBAL] Equity Global

**Total de activos:** 7

---

## 2. Matriz de Covarianza y Análisis de Riesgo Sistémico

### Matriz de Covarianza (Anualizada)

```
Ticker    ALUA.BA   BTC-USD   GGAL.BA     GOOGL        KO      MSFT   YPFD.BA
Ticker                                                                       
ALUA.BA  0.407496 -0.026701  0.151094  0.004711 -0.000164  0.005307  0.198645
BTC-USD -0.026701  0.288247  0.030247  0.018761  0.000291  0.015887  0.002539
GGAL.BA  0.151094  0.030247  0.351948  0.016597 -0.000457  0.013041  0.265200
GOOGL    0.004711  0.018761  0.016597  0.083276 -0.000319  0.030842  0.018284
KO      -0.000164  0.000291 -0.000457 -0.000319  0.020295  0.000370  0.005112
MSFT     0.005307  0.015887  0.013041  0.030842  0.000370  0.050853  0.010255
YPFD.BA  0.198645  0.002539  0.265200  0.018284  0.005112  0.010255  0.385607
```

### Interpretación Económica

La **matriz de covarianza** es fundamental para entender cómo los movimientos de un activo afectan a otro. 
Los valores en la diagonal representan la varianza de cada activo (riesgo individual), mientras que 
los valores fuera de la diagonal muestran la covarianza entre pares de activos.

**Observaciones Clave:**

1. **Activos Argentinos y Riesgo Sistémico:**
   - Los activos argentinos (GGAL.BA, YPFD.BA, ALUA.BA) tienden a presentar covarianzas positivas 
     entre sí, reflejando el **riesgo país** que afecta sistemáticamente al mercado local.
   - Eventos macroeconómicos (inflación, tipo de cambio, política monetaria) impactan 
     simultáneamente a estos activos, incrementando el riesgo sistémico de la cartera.

2. **Diversificación Internacional:**
   - GOOGL presenta covarianzas más bajas con activos argentinos, ofreciendo **beneficios de diversificación**.
   - BTC-USD muestra comportamiento asincrónico, actuando como **activo descorrelacionado**.

3. **Implicaciones para la Gestión de Riesgo:**
   - Una concentración alta en activos argentinos **no reduce el riesgo** por diversificación 
     (correlaciones altas → covarianzas positivas elevadas).
   - La inclusión de activos internacionales **reduce la exposición al riesgo sistémico argentino**.

### Matriz de Correlación

```
Ticker    ALUA.BA   BTC-USD   GGAL.BA     GOOGL        KO      MSFT   YPFD.BA
Ticker                                                                       
ALUA.BA  1.000000 -0.077908  0.398975  0.025571 -0.001800  0.036869  0.501123
BTC-USD -0.077908  1.000000  0.094966  0.121093  0.003803  0.131224  0.007616
GGAL.BA  0.398975  0.094966  1.000000  0.096945 -0.005413  0.097482  0.719883
GOOGL    0.025571  0.121093  0.096945  1.000000 -0.007762  0.473938  0.102033
KO      -0.001800  0.003803 -0.005413 -0.007762  1.000000  0.011505  0.057786
MSFT     0.036869  0.131224  0.097482  0.473938  0.011505  1.000000  0.073233
YPFD.BA  0.501123  0.007616  0.719883  0.102033  0.057786  0.073233  1.000000
```

**Análisis de Correlaciones:**
- Correlaciones > 0.7: Alta dependencia (riesgo de contagio)
- Correlaciones < 0.3: Baja dependencia (buena diversificación)
- Correlaciones negativas: Cobertura natural (hedge)

---

## 3. Carteras Optimizadas

### 3.1 Cartera de Mínima Volatilidad

**Objetivo:** Minimizar el riesgo de la cartera (varianza del portfolio)

**Pesos Óptimos:**
```
GGAL.BA     :   5.25%
YPFD.BA     :  10.94%
GOOGL       :  33.81%
BTC-USD     :  10.00%
MSFT        :  30.00%
KO          :  10.00%

```

**Métricas de Performance:**
- **Retorno Esperado Anualizado:** Calculado mediante media histórica
- **Volatilidad Anualizada:** Riesgo de la cartera (desviación estándar)
- **Sharpe Ratio:** Retorno ajustado por riesgo

**Interpretación:** Esta cartera prioriza la **estabilidad** sobre el retorno, ideal para inversores 
con aversión al riesgo elevada o en contextos de alta incertidumbre macroeconómica.

---

### 3.2 Cartera de Máximo Sharpe Ratio

**Objetivo:** Maximizar el retorno ajustado por riesgo (retorno excedente por unidad de volatilidad)

**Pesos Óptimos:**
```
GGAL.BA     :   6.30%
YPFD.BA     :  19.01%
ALUA.BA     :  20.00%
GOOGL       :  20.26%
BTC-USD     :  10.00%
MSFT        :  14.43%
KO          :  10.00%

```

**Métricas de Performance:**
- **Retorno Esperado Anualizado:** Optimizado para máximo retorno ajustado
- **Volatilidad Anualizada:** Riesgo asumido por la cartera
- **Sharpe Ratio:** Máximo retorno por unidad de riesgo

**Interpretación:** Esta cartera busca la **eficiencia máxima**, ofreciendo el mejor trade-off 
entre riesgo y retorno. Recomendada para inversores con horizonte de mediano a largo plazo.

---

## 4. Análisis de Riesgo: VaR y CVaR

### Metodología

Se utilizó **Simulación de Monte Carlo** con 10,000 iteraciones utilizando **distribución t-Student** 
(en lugar de Normal) para capturar mejor las **"fat tails"** de los mercados emergentes:

- **Value at Risk (VaR):** Pérdida máxima esperada con 95% de confianza
- **Conditional VaR (CVaR):** Pérdida esperada cuando se excede el VaR (tail risk)
- **Distribución:** t-Student (captura eventos extremos mejor que Normal)

### Resultados para Cartera de Máximo Sharpe

Se presentan **DOS ESCENARIOS** para gestión de riesgo:

#### 🔴 ESCENARIO CONSERVADOR - t-Student (df=3)

**Uso recomendado:** Asignación de capital, límites de riesgo, stress testing

```
VaR (1 día, 95%):      -3.22%
CVaR (1 día, 95%):     -5.68%
VaR (anualizado, 95%): -51.12%
CVaR (anualizado, 95%): -90.15%
```

**Interpretación:** Supuesto de **máxima prudencia**. Asume que eventos extremos son más 
frecuentes que lo observado históricamente. Ideal para dimensionar capital de respaldo.

---

#### 🟡 ESCENARIO ESPERADO - t-Student (df=7.0)

**Uso recomendado:** Proyecciones, pricing, análisis comparativo

```
VaR (1 día, 95%):      -2.60%
CVaR (1 día, 95%):     -3.66%
VaR (anualizado, 95%): -41.30%
CVaR (anualizado, 95%): -58.16%
```

**Interpretación:** Basado en grados de libertad **estimados de datos históricos**. 
Refleja el comportamiento observado en el período analizado (2024-2026).

**Método de Estimación:** Maximum Likelihood Estimation (MLE) aplicado a los retornos diarios de la cartera.
La función de verosimilitud maximiza: L(df, μ, σ | datos) para la distribución t-Student.
Estimación obtenida: df ≈ 7.04, donde valores bajos (df < 5) indican mayor presencia de eventos extremos.

---

#### ⚪ BASELINE - Distribución Normal (referencia)

**Uso recomendado:** Solo para comparación académica (NO para gestión de riesgo)

```
VaR (1 día, 95%):      -2.23%
CVaR (1 día, 95%):     -2.89%
VaR (anualizado, 95%): -35.45%
CVaR (anualizado, 95%): -45.82%
```

**⚠️ Advertencia:** La Normal **subestima significativamente** el riesgo en mercados emergentes.

---

### Comparación de Escenarios

| Métrica | Normal | Esperado (df=7.0) | Conservador (df=3) |
|---------|--------|-------------|-------------------|
| VaR (1d) | -2.23% | -2.60% | -3.22% |
| CVaR (1d) | -2.89% | -3.66% | -5.68% |

**Diferencia Conservador vs Normal:**
- VaR: +44.2%
- CVaR: +96.7%

### Interpretación Económica

- **VaR:** En el 95% de los días, la cartera **no perderá más del 3.22%**.
- **CVaR:** En escenarios extremos (5% peor de los casos), la pérdida promedio será del **5.68%**.
- El **CVaR es siempre mayor que el VaR**, capturando el "tail risk" o riesgo de cola.

**¿Por qué t-Student para Argentina?**
- La distribución Normal **subestima** eventos extremos (colas pesadas)
- Argentina tiene historia de crisis recurrentes (2001, 2018, 2019, 2020)
- La t-Student asigna **mayor probabilidad** a pérdidas extremas
- Grados de libertad bajos (df=3) → colas más pesadas → estimación más prudente

**Contexto Argentino:**
Los activos locales contribuyen desproporcionadamente al VaR/CVaR debido a:
- Alta volatilidad macroeconómica
- Riesgo de eventos disruptivos (default, controles cambiarios)  
- Baja liquidez en períodos de estrés
- **Fat tails:** Mayor probabilidad de pérdidas extremas vs mercados desarrollados

---

## 5. Análisis Profundo: Riesgo Sistémico Argentino

### Impacto de la Matriz de Covarianza

La matriz de covarianza revela cómo los activos argentinos están **altamente correlacionados** entre sí:

1. **Diagonal Principal (Varianzas):**
   - Los activos argentinos típicamente muestran varianzas más altas que activos globales
   - Esto refleja la volatilidad inherente del mercado local

2. **Elementos Fuera de la Diagonal (Covarianzas):**
   - Covarianzas positivas altas entre GGAL.BA, YPFD.BA, ALUA.BA indican que se mueven juntos
   - Esto implica que la diversificación entre activos argentinos es **limitada**

3. **Efecto en el Riesgo Total:**
   - La varianza del portfolio es: σ²ₚ = wᵀΣw (donde w son los pesos y Σ la matriz de covarianza)
   - Covarianzas altas incrementan σ²ₚ más que la simple suma de varianzas individuales
   - Este es el **riesgo sistémico no diversificable**

---

## 6. Comparación: Optimización Libre vs Gestionada

### Metodología

Se compararon **dos enfoques de optimización**:

1. **Optimización LIBRE (Sin Restricciones):** Permite cualquier asignación entre 0% y 100% por activo
2. **Optimización GESTIONADA (Con Restricciones):** Aplica límites realistas por tipo de activo:
   - Activos argentinos (.BA): Máximo 20% individual
   - Criptomonedas (-USD): Máximo 10%
   - Activos globales: Mínimo 10-15%, Máximo 30-35%

### Composición de Carteras (Máximo Sharpe)

| Activo | Libre (%) | Gestionada (%) | Diferencia |
|--------|-----------|----------------|------------|
| GGAL.BA | 5.40% | 6.30% | +0.89% |
| YPFD.BA | 16.37% | 19.01% | +2.64% |
| ALUA.BA | 24.76% | 20.00% | -4.76% |
| GOOGL | 17.70% | 20.26% | +2.56% |
| BTC-USD | 22.28% | 10.00% | -12.28% |
| MSFT | 9.36% | 14.43% | +5.07% |
| KO | 4.13% | 10.00% | +5.87% |

### Métricas de Performance

| Métrica | Libre | Gestionada | Diferencia |
|---------|-------|------------|------------|
| **Retorno Anualizado** | 80.73% | 85.51% | +4.78% |
| **Volatilidad Anualizada** | 22.97% | 24.67% | +1.70% |
| **Sharpe Ratio** | 3.30 | 3.26 | -0.03 |

### Métricas de Riesgo (VaR/CVaR Conservador, df=3)

| Métrica | Libre | Gestionada | Mejora |
|---------|-------|------------|--------|
| **VaR (1 día, 95%)** | -2.99% | -3.22% | -0.23% |
| **CVaR (1 día, 95%)** | -5.28% | -5.68% | -0.40% |

### Análisis e Interpretación

**1. Sharpe Ratio:**
   - ✅ Trade-off aceptable: Sharpe disminuyó solo 1.0%

**2. Riesgo de Cola (VaR/CVaR):**
   - El VaR empeoró 0.23 puntos porcentuales
   - CVaR empeoró 0.40 puntos porcentuales

**3. Diversificación:**
   - Libre: 7 activos con peso significativo (>1%)
   - Gestionada: 7 activos con peso significativo (>1%)

**4. Recomendación:**
   - ✅ **USAR CARTERA GESTIONADA:** Trade-off aceptable entre eficiencia y control de riesgo
   - Las restricciones proporcionan mayor robustez y mejor gestión de riesgo de concentración


---

## 7. Conclusiones Técnicas

### Hallazgos Principales

1. **Riesgo Sistémico Elevado:** 
   - La alta correlación entre activos argentinos amplifica el riesgo de la cartera
   - La matriz de covarianza muestra dependencias significativas

2. **Beneficios de Diversificación Internacional:**
   - Activos como GOOGL y BTC-USD reducen el riesgo sistémico argentino
   - La frontera eficiente mejora significativamente con diversificación global

3. **Trade-off Riesgo-Retorno:**
   - La cartera de mínima volatilidad sacrifica retorno por estabilidad
   - La cartera de máximo Sharpe optimiza la eficiencia

### Limitaciones del Modelo

**Supuestos y Restricciones:**
- **Datos históricos:** Performance pasada no garantiza resultados futuros
- **Correlaciones dinámicas:** Las correlaciones pueden cambiar abruptamente en períodos de crisis
- **Costos de optimización:** La optimización no considera costos de transacción (pero sí se incluyen en el backtesting)
- **Supuesto de estacionariedad:** Asume que las estadísticas históricas (media, volatilidad) son representativas del futuro
- **Riesgo de modelo:** La t-Student captura mejor las fat tails que la Normal, pero ningún modelo predice el futuro perfectamente

**Mejoras Implementadas (v2.0):**
- ✅ Uso de distribución **t-Student** para VaR/CVaR (captura eventos extremos)
- ✅ Backtesting con **costos de transacción** reales (comisiones, rebalanceo)
- ✅ Comparación **Active vs Passive** management
- ✅ **Stress Testing** con escenarios extremos predefinidos

---

## 8. Apéndice: Visualizaciones

Ver archivo adjunto: `efficient_frontier.png`

El gráfico incluye:
- Frontera eficiente con 5,000 carteras simuladas
- Carteras óptimas (Mínima Volatilidad y Máximo Sharpe) **con restricciones gestionadas**
- Activos individuales
- Matriz de correlación
- Composición de carteras gestionadas
- Matriz de covarianza

**Nota:** Los gráficos muestran las carteras **GESTIONADAS** (con restricciones), que son las utilizadas 
para el backtesting y stress testing. La comparación con carteras LIBRES (sin restricciones) se encuentra 
en la Sección 6 de este reporte.

---

**Disclaimer:** Este análisis se basa en datos históricos y utiliza **distribución t-Student** para modelar riesgos 
(VaR/CVaR), capturando mejor las "fat tails" que una distribución Normal. Sin embargo, los resultados pasados 
**no garantizan performance futura**. El mercado argentino presenta riesgos específicos (riesgo país, riesgo cambiario, 
riesgo regulatorio, controles de capital) que pueden materializarse súbitamente y de forma no anticipada por modelos 
cuantitativos. La optimización asume estabilidad de correlaciones, lo cual puede no cumplirse en crisis sistémicas. 
**Se recomienda enfáticamente consultar con un asesor financiero certificado** antes de tomar decisiones de inversión 
y considerar la tolerancia al riesgo personal, horizonte temporal y situación financiera individual.

---

*Generado por Portfolio Engine v2.0 | Python + Scipy*  
*Desarrollado por Jorge Iván Juárez A. - Lic. en Economía especializado en mercado de capitales*


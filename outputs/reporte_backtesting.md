# Reporte de Backtesting - Validación Histórica
## Análisis Ex-Post con Gestión Activa vs Pasiva

**Fecha de Generación:** 2026-02-17 19:58:57  
**Período de Optimización (Train):** 2023-02-18 a 2025-02-17  
**Período de Backtest (Test):** 2025-02-17 a 2026-02-17  
**Estrategia Validada:** Cartera de Máximo Sharpe Ratio **GESTIONADA** (con restricciones)  
**Capital Inicial:** $1,000,000.00 USD  
**Comisión por Operación:** 0.5%

> **📋 Nota sobre la estrategia:** Se validó la cartera **GESTIONADA** (con restricciones por tipo de activo) 
> en lugar de la LIBRE (sin restricciones), ya que la gestionada ofrece mejor control de riesgo de concentración 
> con un trade-off mínimo de eficiencia. Ver **Sección 6** del Reporte de Optimización para la comparación completa.

✅ **Validación Out-of-Sample Genuina**

> El backtesting usa datos **posteriores** al período de optimización, proporcionando una validación robusta.

---

## RESUMEN EJECUTIVO

La estrategia fue **validada históricamente** comparando **DOS ENFOQUES**:

1. **ACTIVA (Rebalanceo Mensual):** Ajusta pesos al target cada mes, pagando comisiones
2. **PASIVA (Buy-and-Hold):** Compra inicial sin rebalanceo, comisión única

### Resultados Comparativos

| Métrica | Active (Rebalanceo) | Passive (Buy-Hold) | Diferencia |
|---------|--------------------|--------------------|------------|
| **Capital Final** | $1,233,969.85 | $1,232,520.22 | $+1,449.63 |
| **Retorno Total** | +23.40% | +23.25% | +0.14% |
| **Retorno Anualizado** | +25.53% | +25.37% | +0.16% |
| **Sharpe Ratio** | 0.84 | 0.80 | +0.05 |
| **Máximo Drawdown** | -15.03% | -15.00% | -0.03% |
| **Comisiones Totales** | $10,077.56 (1.01%) | $5,000.00 (0.50%) | $5,077.56 |

### CONCLUSION CLAVE: REBALANCEO NO VALIO LA PENA

El rebalanceo activo NO justificó las comisiones adicionales de $5,077.56. La estrategia pasiva fue superior en 0.14496303671065291 puntos porcentuales.

### Comparación vs Benchmark (SPY)

| Métrica | Active | Passive | Benchmark | Active vs Bench | Passive vs Bench |
|---------|--------|---------|-----------|-----------------|------------------|
| Retorno Total | +23.40% | +23.25% | +12.24% | +11.16% MEJOR | +11.02% MEJOR |
| Capital Final | $1,233,969.85 | $1,232,520.22 | $1,122,352.16 | - | - |
| Max Drawdown | -15.03% | -15.00% | -18.76% | Mejor | Mejor |

---

## 1. Metodología del Backtest

### 🎯 Validación Out-of-Sample

**Este backtesting utiliza una metodología rigurosa de validación out-of-sample:**

- **Train Set (Optimización):** Los pesos óptimos se calcularon usando datos históricos **previos** al período de backtesting
- **Test Set (Backtesting):** La validación usa datos **posteriores** que el modelo **nunca vio** durante la optimización
- **Objetivo:** Evitar data leakage y overfitting, proporcionando una estimación realista del performance futuro

**¿Por qué es importante?**
- ✅ **Honestidad metodológica:** No "miramos al futuro" para optimizar
- ✅ **Estimación realista:** Los resultados reflejan el performance en datos nuevos
- ✅ **Previene overfitting:** El modelo no está "sobreajustado" a los datos de validación

---

### Tipo de Backtest
- **Estrategias:** 
  - **ACTIVA:** Rebalanceo mensual a pesos target + comisiones 0.5% por volumen operado
  - **PASIVA:** Buy & Hold sin rebalanceo + comisión inicial única 0.5%
- **Período:** 1 año (2025-02-17 → 2026-02-17)
- **Capital Inicial:** $1,000,000.00 USD
- **Frecuencia:** Diaria (ajustado al cierre)
- **Costos:** Comisiones de 0.5% por operación (realista)

### Composición de la Cartera Testeada

**Pesos de la Cartera de Máximo Sharpe GESTIONADA:**

```
GGAL.BA     :   6.30%
YPFD.BA     :  19.01%
ALUA.BA     :  20.00%
GOOGL       :  20.26%
BTC-USD     :  10.00%
MSFT        :  14.43%
KO          :  10.00%

```

**Restricciones aplicadas:**
- Activos argentinos (.BA): Máximo 20% individual
- Criptomonedas (-USD): Máximo 10%
- Activos globales: Mínimo 10-15%, Máximo 30-35%

### Benchmark
- **Índice de Referencia:** SPY (S&P 500 ETF)
- **Razón:** Proxy del mercado estadounidense para comparación con estrategia global

---

## 2. Resultados de Performance

### 2.1 Performance Absoluta

#### Capital Acumulado
```
Capital Inicial:       $1,000,000.00
Capital Final:         $1,233,969.85
Ganancia/Pérdida:      $233,969.85 (+23.40%)
```

#### Retornos
```
Retorno Total (período):   +23.40%
Retorno Anualizado:        +25.53%
```

**Interpretación:**
- La cartera generó retornos positivos, cumpliendo con las expectativas de la optimización.
- El retorno anualizado supera el típico rendimiento de bonos soberanos (5-7%), validando la estrategia.

### 2.2 Métricas de Riesgo Realizadas

#### Volatilidad
```
Volatilidad Anualizada:    25.47%
```

**Interpretación:**
- Volatilidad moderada, adecuada para el perfil de riesgo de la cartera.

#### Sharpe Ratio Realizado
```
Sharpe Ratio:              0.84
```

**Benchmarks de Sharpe:**
- < 0: Estrategia destruye valor
- 0 - 1: Retorno no compensa el riesgo adecuadamente
- 1 - 2: Buena relación riesgo-retorno ✅
- 2+: Excelente relación riesgo-retorno ⭐

**Veredicto:** ⚠️ La relación riesgo-retorno es subóptima

---

## 3. Análisis de Drawdown (Caídas)

### Máximo Drawdown Histórico

```
Máximo Drawdown:           -15.03%
```

**¿Qué significa?**
El Máximo Drawdown (MDD) representa la **caída más profunda** que experimentó la cartera 
desde un máximo histórico hasta un mínimo posterior. Es una medida crítica de **riesgo 
de pérdida temporal**.

**Interpretación:**
- Drawdown controlado, dentro de límites aceptables para inversores moderados.
- Este nivel de caída es típico en carteras con exposición a mercados emergentes y cripto.

### Comparación con VaR/CVaR Proyectado

Recordemos los niveles de riesgo proyectados ex-ante:

| Escenario | VaR (1 día, 95%) | CVaR (1 día, 95%) | VaR Anualizado | CVaR Anualizado |
|-----------|------------------|-------------------|----------------|-----------------|
| Conservador (df=3) | -3.22% | -5.68% | -51.12% | -90.15% |
| Esperado (df=7.0) | -2.60% | -3.66% | -41.30% | -58.16% |
| Normal (baseline) | -2.23% | -2.89% | -35.45% | -45.82% |

**Máximo Drawdown Realizado:** -15.03%

**Análisis Comparativo:**
✅ **El drawdown realizado (15.03%) fue MENOR que el VaR conservador proyectado (51.12%).**

Esto indica que:
- El modelo de VaR fue **prudente y adecuado**
- No se materializaron eventos extremos en el período
- La cartera se comportó dentro de los parámetros esperados

---

## 4. Comparación: Cartera vs Benchmark

### 4.1 Retornos

```
Estrategia Activa:         +23.40%
Estrategia Pasiva:         +23.25%
Benchmark (SPY):           +12.24%
Active vs Bench:           +11.16% MEJOR
Passive vs Bench:          +11.02% MEJOR
```

**Interpretación:**
✅ **La estrategia activa SUPERÓ al benchmark** en 11.16 puntos porcentuales. El rebalanceo agregó valor.

✅ **La estrategia pasiva superó al benchmark** en 11.02%.

### 4.2 Riesgo (Drawdown)

```
Cartera:                   -15.03%
Benchmark:                 -18.76%
```

**Interpretación:**
✅ La cartera tuvo **menor drawdown** que el benchmark, mostrando mejor gestión de riesgo.

---

## 5. Validación de Supuestos: Proyectado vs Realizado

### 5.1 Retorno Esperado vs Retorno Realizado

En el análisis ex-ante se proyectó un retorno anualizado basado en datos históricos (2.0 años). 
El backtest nos permite validar si esas proyecciones fueron precisas.

**Resultado:** ✅ El retorno realizado (25.53%) está alineado con las proyecciones.

### 5.2 Volatilidad Proyectada vs Volatilidad Realizada

La volatilidad anualizada realizada fue de **25.47%**.

**Análisis:** La volatilidad realizada está dentro del rango esperado para esta cartera.

### 5.3 VaR/CVaR: ¿Fue Preciso?

El VaR y CVaR son medidas prospectivas de riesgo. El backtest nos permite verificar si 
los modelos fueron adecuados:

- **VaR Conservador (anual):** -51.12%
- **Máximo Drawdown Realizado:** -15.03%

✅ El VaR conservador fue ADECUADO: el drawdown real (15.03%) fue menor al VaR proyectado (51.12%).

---

## 6. Visualizaciones

Ver archivo adjunto: **`backtest_results.png`**

El gráfico incluye:

1. **Equity Curve:** Evolución del capital de la cartera vs benchmark
2. **Drawdown:** Caídas desde máximos históricos
3. **Distribución de Retornos Diarios:** Histograma de retornos
4. **Comparación de Métricas:** Tabla visual de performance

---

## 7. Conclusiones y Lecciones Aprendidas

### ✅ Fortalezas de la Estrategia

- **Retornos Positivos:** La cartera generó ganancias en el período analizado
- **Outperformance vs Benchmark:** La estrategia activa superó al S&P 500 en 11.16%

### ⚠️ Debilidades Identificadas

- No se identificaron debilidades críticas en el período analizado

### 🔍 Lecciones Aprendidas

1. **Validación de Modelos:** 
   - El VaR conservador fue apropiado para gestión de riesgo.
   
2. **Comportamiento en Crisis:**
   - El drawdown máximo (-15.03%) muestra la resiliencia de la cartera en períodos adversos
   
3. **Diversificación:**
   - La combinación de activos argentinos, globales y cripto cumplió su función de reducir riesgo

---

## 8. Recomendaciones para Implementación

### Para Inversores Conservadores
- Considerar **reducir exposición a activos argentinos** si el drawdown supera tolerancia
- Implementar **stop-loss** en nivel cercano al VaR diario (-3.22%)
- **Rebalancear** trimestralmente para mantener pesos óptimos

### Para Inversores Agresivos
- La estrategia requiere ajustes en la asignación
- Considerar **apalancamiento moderado** si Sharpe > 2
- Monitorear **indicadores macro argentinos** (riesgo país, tipo de cambio)

### Ajustes Sugeridos

2. **Revisar la composición de la cartera** - el riesgo no está siendo compensado adecuadamente

---

## 9. Próximos Pasos

1. **Backtest Rolling (ventana móvil):** Evaluar estabilidad de la estrategia en diferentes períodos
2. **Out-of-Sample Testing:** Testear en datos más recientes no usados en optimización
3. **Stress Testing:** Simular escenarios extremos (crisis 2001, 2018, pandemia 2020)
4. **Optimización Dinámica:** Implementar rebalanceo mensual/trimestral
5. **Inclusión de Costos:** Agregar comisiones y slippage para análisis realista

---

## Ver También

- **📈 Análisis de Optimización (Ex-Ante):** Consultar `reporte_portfolio.md`
- **📊 Gráficos:**
  - Frontera Eficiente: `efficient_frontier.png`
  - Resultados del Backtest: `backtest_results.png`

---

**Disclaimer:** Este backtest se basa en datos históricos y supone un escenario ideal sin costos 
de transacción. Los resultados pasados no garantizan performance futura. Las condiciones de 
mercado pueden cambiar dramáticamente, especialmente en mercados emergentes como Argentina. 
Se recomienda consultar con un asesor financiero certificado antes de implementar esta estrategia.

---

*Generado por Portfolio Engine v2.0 | Python + Scipy*  
*Desarrollado por Jorge Iván Juárez A. - Lic. en Economía especializado en mercado de capitales*

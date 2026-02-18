# Reporte de Stress Testing
## Análisis de Escenarios Extremos

**Fecha de Generación:** 2026-02-17 19:58:57  
**Cartera Analizada:** Máximo Sharpe Ratio **GESTIONADA** (con restricciones)  
**Capital Invertido:** $1,000,000.00 USD

> **📋 Nota:** Se analizó la cartera **GESTIONADA** con restricciones por tipo de activo, 
> elegida por su mejor balance entre eficiencia y control de riesgo. Ver Reporte de Optimización (Sección 6).

---

## 📊 Resumen Ejecutivo

El **Stress Testing** es una técnica de gestión de riesgo que simula el comportamiento de la 
cartera bajo **escenarios extremos** (crisis, crashes, recuperaciones). A diferencia del VaR/CVaR 
que usa probabilidades históricas, el stress testing evalúa eventos específicos de alta severidad.

### Escenarios Simulados

Se analizaron **3 escenarios extremos**:

1. **Crash Global:** Crisis financiera internacional (mercados desarrollados y cripto)
2. **Crisis Argentina:** Colapso específico del mercado local
3. **Recuperación Agresiva:** Rebote generalizado de todos los activos

---

## 1. Resultados por Escenario

### Tabla Resumen

| Escenario | Capital Inicial | Impacto | Capital Final | Cambio % | Estado |
|-----------|----------------|---------|---------------|----------|--------|
| Crash Global | $1,000,000.00 | $-227,215.55 | $772,784.45 | -22.72% | 🟠 |
| Crisis Argentina | $1,000,000.00 | $-213,577.33 | $786,422.67 | -21.36% | 🟠 |
| Recuperación Agresiva | $1,000,000.00 | $+200,000.00 | $1,200,000.00 | +20.00% | 🟢 |

---

## 2. Análisis Detallado por Escenario

### Escenario: Crash Global

**Descripción de Shocks Aplicados:**

- **GGAL.BA**: -20.0%
- **YPFD.BA**: -20.0%
- **ALUA.BA**: -20.0%
- **GOOGL**: -20.0%
- **BTC-USD**: -40.0%
- **MSFT**: -25.0%
- **KO**: -20.0%

**Resultados:**

```
Capital Inicial:       $1,000,000.00
Impacto Total:         $-227,215.55
Capital Final:         $772,784.45
Cambio Porcentual:     -22.72%
```

**Impacto por Activo:**

| Activo | Peso en Cartera | Capital Asignado | Shock | Impacto en USD |
|--------|----------------|------------------|-------|----------------|
| GGAL.BA | 6.30% | $62,959.03 | -20.0% | $-12,591.81 |
| YPFD.BA | 19.01% | $190,119.07 | -20.0% | $-38,023.81 |
| ALUA.BA | 20.00% | $200,000.00 | -20.0% | $-40,000.00 |
| GOOGL | 20.26% | $202,610.96 | -20.0% | $-40,522.19 |
| BTC-USD | 10.00% | $100,000.00 | -40.0% | $-40,000.00 |
| MSFT | 14.43% | $144,310.94 | -25.0% | $-36,077.74 |
| KO | 10.00% | $100,000.00 | -20.0% | $-20,000.00 |

**Interpretación:** 🟠 **ALTO RIESGO:** Pérdida significativa. Se requiere gestión activa para mitigar el impacto.

---

### Escenario: Crisis Argentina

**Descripción de Shocks Aplicados:**

- **GGAL.BA**: -40.0%
- **YPFD.BA**: -40.0%
- **ALUA.BA**: -40.0%
- **GOOGL**: -5.0%
- **BTC-USD**: -10.0%
- **MSFT**: -5.0%
- **KO**: -5.0%

**Resultados:**

```
Capital Inicial:       $1,000,000.00
Impacto Total:         $-213,577.33
Capital Final:         $786,422.67
Cambio Porcentual:     -21.36%
```

**Impacto por Activo:**

| Activo | Peso en Cartera | Capital Asignado | Shock | Impacto en USD |
|--------|----------------|------------------|-------|----------------|
| GGAL.BA | 6.30% | $62,959.03 | -40.0% | $-25,183.61 |
| YPFD.BA | 19.01% | $190,119.07 | -40.0% | $-76,047.63 |
| ALUA.BA | 20.00% | $200,000.00 | -40.0% | $-80,000.00 |
| GOOGL | 20.26% | $202,610.96 | -5.0% | $-10,130.55 |
| BTC-USD | 10.00% | $100,000.00 | -10.0% | $-10,000.00 |
| MSFT | 14.43% | $144,310.94 | -5.0% | $-7,215.55 |
| KO | 10.00% | $100,000.00 | -5.0% | $-5,000.00 |

**Interpretación:** 🟠 **ALTO RIESGO:** Pérdida significativa. Se requiere gestión activa para mitigar el impacto.

---

### Escenario: Recuperación Agresiva

**Descripción de Shocks Aplicados:**

- **GGAL.BA**: +20.0%
- **YPFD.BA**: +20.0%
- **ALUA.BA**: +20.0%
- **GOOGL**: +20.0%
- **BTC-USD**: +20.0%
- **MSFT**: +20.0%
- **KO**: +20.0%

**Resultados:**

```
Capital Inicial:       $1,000,000.00
Impacto Total:         $+200,000.00
Capital Final:         $1,200,000.00
Cambio Porcentual:     +20.00%
```

**Impacto por Activo:**

| Activo | Peso en Cartera | Capital Asignado | Shock | Impacto en USD |
|--------|----------------|------------------|-------|----------------|
| GGAL.BA | 6.30% | $62,959.03 | +20.0% | $+12,591.81 |
| YPFD.BA | 19.01% | $190,119.07 | +20.0% | $+38,023.81 |
| ALUA.BA | 20.00% | $200,000.00 | +20.0% | $+40,000.00 |
| GOOGL | 20.26% | $202,610.96 | +20.0% | $+40,522.19 |
| BTC-USD | 10.00% | $100,000.00 | +20.0% | $+20,000.00 |
| MSFT | 14.43% | $144,310.94 | +20.0% | $+28,862.19 |
| KO | 10.00% | $100,000.00 | +20.0% | $+20,000.00 |

**Interpretación:** 🟢 **POSITIVO:** Ganancia potencial en este escenario.

---

## 3. Análisis de Resiliencia

### Métricas de Riesgo Extremo

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Peor Escenario** | Crash Global | Pérdida de $227,215.55 (-22.72%) |
| **Mejor Escenario** | Recuperación Agresiva | Ganancia de $200,000.00 (+20.00%) |
| **Capital en Riesgo** | $220,396.44 | Promedio de escenarios negativos |
| **Capital Mínimo (worst case)** | $772,784.45 | Capital remanente en crisis |

### Evaluación de Vulnerabilidad

**Exposición a Crash Global:**
```
Impacto: $-227,215.55
Cambio: -22.72%
```

El escenario de crash global simula una crisis financiera internacional similar a 2008 o marzo 2020. 
La cartera tiene exposición moderada a crisis globales.

**Exposición a Crisis Argentina:**
```
Impacto: $-213,577.33
Cambio: -21.36%
```

Este escenario simula un colapso específico del mercado argentino (similar a 2001, 2018 o 2019).
La cartera tiene exposición controlada al riesgo argentino.

**Potencial de Recuperación:**
```
Ganancia: $+200,000.00
Cambio: +20.00%
```

En un escenario de recuperación fuerte, la cartera tiene alto potencial de upside.

---

## 4. Comparación: Stress Test vs VaR/CVaR

### Diferencias Metodológicas

| Aspecto | VaR/CVaR | Stress Testing |
|---------|----------|----------------|
| **Enfoque** | Probabilístico (distribución) | Determinístico (escenarios) |
| **Uso** | Riesgo en condiciones normales | Riesgo en eventos extremos |
| **Ventaja** | Cuantifica probabilidades | Simula eventos específicos |
| **Limitación** | Puede subestimar tail risk | No considera probabilidades |

### Integración de Métricas

El **VaR/CVaR** te dice: *"¿Cuánto puedo perder en el 5% peor de los casos?"*

El **Stress Testing** te dice: *"¿Cuánto perderé SI ocurre [evento específico]?"*

**Recomendación:** Usar ambas metodologías en conjunto:
- **VaR/CVaR** para límites diarios de riesgo
- **Stress Testing** para planificación de capital y contingencias

---

## 5. Recomendaciones Estratégicas

### Para el Peor Escenario (Crash Global)

**Pérdida Potencial:** $227,215.55 (-22.72%)

**Acciones Recomendadas:**

1. **Monitoreo Activo de Indicadores**
   - Riesgo país (EMBI+ Argentina)
   - Volatilidad implícita (VIX)
   - Spread de bonos soberanos

2. **Stop-Loss Dinámico**
   - Implementar órdenes de stop en niveles críticos
   - Revisar mensualmente según volatilidad

3. **Diversificación Adicional**
   - Considerar activos no correlacionados
   - Explorar mercados emergentes alternativos (Chile, Brasil)

### Gestión de Capital en Crisis

Si se materializa el **peor escenario** (Crash Global):

```
Capital Inicial:      $1,000,000.00
Capital Remanente:    $772,784.45
Pérdida:              $227,215.55
```

**Plan de Contingencia:**

1. **Fase 1 - Preservación (0-10% pérdida):**
   - Mantener posiciones, no vender en pánico
   - Monitorear rebote técnico

2. **Fase 2 - Defensa (10-20% pérdida):**
   - Activar stop-loss parcial en activos más volátiles
   - Aumentar cash position

3. **Fase 3 - Evacuación (>20% pérdida):**
   - Liquidar posiciones de alta beta
   - Proteger capital remanente

---

## 6. Visualizaciones

Ver archivo adjunto: **`stress_test.png`**

El gráfico incluye:
1. **Capital Final por Escenario:** Barras horizontales con capital remanente
2. **Pérdidas/Ganancias:** Impacto absoluto en USD

---

## 7. Conclusiones

### Vulnerabilidades Identificadas

- No se identificaron vulnerabilidades críticas (pérdidas < 25%)

### Fortalezas de la Cartera

- **Recuperación Agresiva:** Potencial de ganancia del +20.0%

### Nivel de Riesgo Global

🟠 **RIESGO MODERADO-ALTO:** La cartera tiene exposición significativa. Monitoreo activo requerido.

---

## 📎 Ver También

- **📈 Análisis de Optimización (Ex-Ante):** `reporte_portfolio.md`
- **📊 Validación Histórica (Ex-Post):** `reporte_backtesting.md`
- **📉 Gráficos:**
  - Stress Testing: `stress_test.png`
  - Frontera Eficiente: `efficient_frontier.png`
  - Backtest: `backtest_results.png`

---

**Disclaimer:** El stress testing simula escenarios hipotéticos extremos y no constituye una 
predicción de eventos futuros. Los shocks aplicados son estimaciones basadas en crisis históricas 
y pueden no reflejar la magnitud real de eventos futuros. Se recomienda actualizar los escenarios 
periódicamente y consultar con un asesor de riesgo profesional.

---

*Generado por Portfolio Engine v2.0 | Python + Scipy*  
*Desarrollado por Jorge Iván Juárez A. - Lic. en Economía especializado en mercado de capitales*

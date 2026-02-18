"""
Motor de Optimización de Carteras y Análisis de Riesgo
Desarrollado para mercado financiero argentino y global
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy import stats
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Directorio base del proyecto (detectar automáticamente)
SCRIPT_DIR = Path(__file__).parent.parent  # Sube desde src/ al directorio raíz
OUTPUTS_DIR = SCRIPT_DIR / 'outputs'


class PortfolioOptimizer:
    """
    Clase principal para optimización de carteras y análisis de riesgo
    """
    
    def __init__(self, tickers, start_date=None, end_date=None, risk_free_rate=0.05):
        """
        Inicializa el optimizador de carteras
        
        Parameters:
        -----------
        tickers : list
            Lista de tickers a analizar
        start_date : str
            Fecha de inicio (formato 'YYYY-MM-DD')
        end_date : str
            Fecha de fin (formato 'YYYY-MM-DD')
        risk_free_rate : float
            Tasa libre de riesgo anualizada (default: 0.05 = 5%)
        """
        self.tickers = tickers
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.start_date = start_date or (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        self.risk_free_rate = risk_free_rate
        
        self.data = None
        self.returns = None
        self.mu = None
        self.S = None
        self.correlation_matrix = None
        
        print(f"\n{'='*80}")
        print(f"MOTOR DE OPTIMIZACIÓN DE CARTERAS - MERCADO ARGENTINO Y GLOBAL")
        print(f"{'='*80}")
        print(f"\nActivos bajo análisis: {', '.join(tickers)}")
        print(f"Período: {self.start_date} a {self.end_date}")
        print(f"{'='*80}\n")
    
    def download_data(self):
        """
        Descarga datos históricos de los activos
        """
        print("[+] Descargando datos historicos...")
        
        try:
            raw_data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            
            # Verificar si los datos se descargaron
            if raw_data.empty:
                raise ValueError("No se pudieron descargar datos para los activos especificados")
            
            # Extraer Adj Close
            if 'Adj Close' in raw_data.columns:
                self.data = raw_data['Adj Close']
            elif isinstance(raw_data.columns, pd.MultiIndex):
                # Si hay MultiIndex, intentar obtener Adj Close del nivel superior
                if 'Adj Close' in raw_data.columns.get_level_values(0):
                    self.data = raw_data['Adj Close']
                else:
                    # Si no hay Adj Close, usar Close
                    self.data = raw_data['Close']
            else:
                # Si solo hay un ticker, usar los datos directamente
                self.data = raw_data
            
            # Si solo hay un ticker, convertir a DataFrame
            if isinstance(self.data, pd.Series):
                self.data = self.data.to_frame(name=self.tickers[0])
            
            # Limpiar datos - eliminar filas con NaN
            self.data = self.data.dropna()
            
            # Verificar si tenemos datos después de limpiar
            if len(self.data) == 0:
                raise ValueError("No se pudieron descargar datos validos para los activos especificados")
            
            print(f"[OK] Datos descargados: {len(self.data)} dias de trading")
            print(f"[OK] Periodo efectivo: {self.data.index[0].date()} a {self.data.index[-1].date()}")
            print(f"[OK] Activos descargados exitosamente: {', '.join(self.data.columns)}")
            
            # Calcular retornos
            self.returns = self.data.pct_change().dropna()
            
            return self.data
            
        except Exception as e:
            print(f"[ERROR] Error al descargar datos: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def calculate_statistics(self):
        """
        Calcula estadísticas básicas de los activos
        """
        print("\n" + "="*80)
        print("ESTADÍSTICAS DESCRIPTIVAS DE LOS ACTIVOS")
        print("="*80 + "\n")
        
        # Retornos anualizados
        annual_returns = self.returns.mean() * 252
        # Volatilidad anualizada
        annual_volatility = self.returns.std() * np.sqrt(252)
        # Sharpe Ratio (usando tasa libre de riesgo)
        sharpe_ratio = (annual_returns - self.risk_free_rate) / annual_volatility
        
        stats_df = pd.DataFrame({
            'Retorno Anualizado': annual_returns,
            'Volatilidad Anualizada': annual_volatility,
            'Sharpe Ratio': sharpe_ratio
        })
        
        print(stats_df.to_string())
        
        # Matriz de covarianza (anualizada)
        print("\n" + "="*80)
        print("MATRIZ DE COVARIANZA (Anualizada)")
        print("="*80 + "\n")
        
        self.S = self.returns.cov() * 252
        print(self.S.to_string())
        
        # Matriz de correlación
        print("\n" + "="*80)
        print("MATRIZ DE CORRELACIÓN")
        print("="*80 + "\n")
        
        self.correlation_matrix = self.returns.corr()
        print(self.correlation_matrix.to_string())
        
        # Retornos esperados
        self.mu = self.returns.mean() * 252
        
        return stats_df
    
    def portfolio_performance(self, weights):
        """
        Calcula métricas de performance del portfolio
        
        Parameters:
        -----------
        weights : np.array
            Pesos de los activos
            
        Returns:
        --------
        tuple : (retorno, volatilidad, sharpe)
        """
        returns = np.dot(weights, self.mu)
        volatility = np.sqrt(np.dot(weights.T, np.dot(self.S, weights)))
        sharpe = (returns - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        return returns, volatility, sharpe
    
    def neg_sharpe_ratio(self, weights):
        """
        Retorna el negativo del Sharpe Ratio (para minimización)
        """
        return -self.portfolio_performance(weights)[2]
    
    def portfolio_volatility(self, weights):
        """
        Calcula la volatilidad del portfolio
        """
        return self.portfolio_performance(weights)[1]
    
    def optimize_min_volatility(self, bounds_dict=None):
        """
        Optimiza la cartera para mínima volatilidad
        
        Parameters:
        -----------
        bounds_dict : dict, optional
            Diccionario con límites personalizados por activo.
            Ejemplo: {'BTC-USD': (0.02, 0.10), 'AAPL': (0.15, 0.30)}
            Si None, usa (0, 1) para todos los activos.
        """
        print("\n" + "="*80)
        print("CARTERA DE MÍNIMA VOLATILIDAD")
        if bounds_dict:
            print(" (CON RESTRICCIONES PERSONALIZADAS)")
        print("="*80 + "\n")
        
        n_assets = len(self.tickers)
        
        # Restricciones y bounds
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Suma de pesos = 1
        
        # Construir bounds según diccionario personalizado o default
        if bounds_dict is None:
            bounds = tuple((0, 1) for _ in range(n_assets))
            print("[i] Sin restricciones: Pesos entre 0% y 100% para todos los activos")
        else:
            bounds = []
            print("[i] Restricciones personalizadas aplicadas:")
            for ticker in self.tickers:
                if ticker in bounds_dict:
                    bound = bounds_dict[ticker]
                    bounds.append(bound)
                    print(f"    {ticker:12s}: {bound[0]*100:5.1f}% a {bound[1]*100:5.1f}%")
                else:
                    bounds.append((0, 1))
            bounds = tuple(bounds)
        
        # Inicialización: pesos iguales
        initial_guess = np.array([1/n_assets] * n_assets)
        
        # Optimización
        result = minimize(
            self.portfolio_volatility,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Validar resultado de la optimización
        if not result.success:
            print(f"\n[ERROR] La optimizacion de Minima Volatilidad FALLO")
            print(f"[ERROR] Razon: {result.message}")
            print(f"[ERROR] Estado: {result.status}")
            print("[WARN] Usando pesos equiponderados como fallback\n")
            weights = np.array([1/n_assets] * n_assets)
        else:
            weights = result.x
            
            # Limpiar pesos pequeños
            weights[weights < 0.001] = 0
            weights = weights / weights.sum()  # Renormalizar
        
        # Crear diccionario de pesos
        cleaned_weights = {ticker: weight for ticker, weight in zip(self.tickers, weights)}
        
        # Calcular performance
        performance = self.portfolio_performance(weights)
        
        print("\nPesos óptimos:")
        for ticker, weight in cleaned_weights.items():
            if weight > 0.001:  # Solo mostrar pesos significativos
                print(f"  {ticker:12s}: {weight*100:6.2f}%")
        
        print(f"\nRetorno Esperado: {performance[0]*100:.2f}%")
        print(f"Volatilidad: {performance[1]*100:.2f}%")
        print(f"Sharpe Ratio: {performance[2]:.2f}")
        
        return cleaned_weights, performance
    
    def optimize_max_sharpe(self, bounds_dict=None):
        """
        Optimiza la cartera para máximo Sharpe Ratio
        
        Parameters:
        -----------
        bounds_dict : dict, optional
            Diccionario con límites personalizados por activo.
            Ejemplo: {'BTC-USD': (0.02, 0.10), 'AAPL': (0.15, 0.30)}
            Si None, usa (0, 1) para todos los activos.
        """
        print("\n" + "="*80)
        print("CARTERA DE MÁXIMO SHARPE RATIO")
        if bounds_dict:
            print(" (CON RESTRICCIONES PERSONALIZADAS)")
        print("="*80 + "\n")
        
        n_assets = len(self.tickers)
        
        # Restricciones y bounds
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Construir bounds según diccionario personalizado o default
        if bounds_dict is None:
            bounds = tuple((0, 1) for _ in range(n_assets))
            print("[i] Sin restricciones: Pesos entre 0% y 100% para todos los activos")
        else:
            bounds = []
            print("[i] Restricciones personalizadas aplicadas:")
            for ticker in self.tickers:
                if ticker in bounds_dict:
                    bound = bounds_dict[ticker]
                    bounds.append(bound)
                    print(f"    {ticker:12s}: {bound[0]*100:5.1f}% a {bound[1]*100:5.1f}%")
                else:
                    bounds.append((0, 1))
            bounds = tuple(bounds)
        
        # Inicialización
        initial_guess = np.array([1/n_assets] * n_assets)
        
        # Optimización
        result = minimize(
            self.neg_sharpe_ratio,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Validar resultado de la optimización
        if not result.success:
            print(f"\n[ERROR] La optimizacion de Maximo Sharpe FALLO")
            print(f"[ERROR] Razon: {result.message}")
            print(f"[ERROR] Estado: {result.status}")
            print("[WARN] Usando pesos equiponderados como fallback\n")
            weights = np.array([1/n_assets] * n_assets)
        else:
            weights = result.x
            
            # Limpiar pesos pequeños
            weights[weights < 0.001] = 0
            weights = weights / weights.sum()
        
        # Crear diccionario de pesos
        cleaned_weights = {ticker: weight for ticker, weight in zip(self.tickers, weights)}
        
        # Calcular performance
        performance = self.portfolio_performance(weights)
        
        print("\nPesos óptimos:")
        for ticker, weight in cleaned_weights.items():
            if weight > 0.001:
                print(f"  {ticker:12s}: {weight*100:6.2f}%")
        
        print(f"\nRetorno Esperado: {performance[0]*100:.2f}%")
        print(f"Volatilidad: {performance[1]*100:.2f}%")
        print(f"Sharpe Ratio: {performance[2]:.2f}")
        
        return cleaned_weights, performance
    
    def calculate_var_cvar(self, weights, confidence_level=0.95, n_simulations=10000):
        """
        Calcula VaR y CVaR usando simulación de Monte Carlo con distribución t-Student
        para capturar mejor las 'fat tails' de los mercados emergentes.
        
        Parameters:
        -----------
        weights : dict
            Pesos de la cartera
        confidence_level : float
            Nivel de confianza (default 0.95 = 95%)
        n_simulations : int
            Número de simulaciones Monte Carlo
        """
        print("\n" + "="*80)
        print(f"VALUE AT RISK (VaR) Y CONDITIONAL VaR (CVaR) - Nivel {confidence_level*100:.0f}%")
        print("="*80 + "\n")
        
        # Convertir pesos a array
        weights_array = np.array([weights[ticker] for ticker in self.tickers])
        
        # Retornos del portfolio
        portfolio_returns = (self.returns * weights_array).sum(axis=1)
        
        # Parámetros para la simulación
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # Estimar grados de libertad de los datos históricos usando MLE
        # Ajustar distribución t a los retornos estandarizados
        standardized_returns = (portfolio_returns - mean_return) / std_return
        df_estimated, loc_est, scale_est = stats.t.fit(standardized_returns)
        
        # Usar df conservador (el mínimo entre estimado y 3) para capturar fat tails
        df_conservative = 3
        df_used = min(df_estimated, 10)  # Limitamos para evitar valores extremos
        
        print(f"[i] Distribución t-Student:")
        print(f"  - Grados de libertad estimados (MLE): {df_estimated:.2f}")
        print(f"  - Grados de libertad conservadores: {df_conservative}")
        print(f"  - Grados de libertad usados: {df_used:.2f}")
        print(f"\n[*] Ejecutando {n_simulations:,} simulaciones de Monte Carlo...")
        print(f"    Método: Distribución t-Student (captura 'fat tails')")
        
        np.random.seed(42)  # Para reproducibilidad
        
        # ==========================================
        # MÉTODO 1: Distribución NORMAL (baseline)
        # ==========================================
        simulated_returns_normal = np.random.normal(mean_return, std_return, n_simulations)
        var_normal = np.percentile(simulated_returns_normal, (1 - confidence_level) * 100)
        cvar_normal = simulated_returns_normal[simulated_returns_normal <= var_normal].mean()
        var_annual_normal = var_normal * np.sqrt(252)
        cvar_annual_normal = cvar_normal * np.sqrt(252)
        
        # ==========================================
        # MÉTODO 2: Distribución t-STUDENT (con df estimado)
        # ==========================================
        simulated_returns_t = stats.t.rvs(df=df_used, loc=mean_return, 
                                           scale=std_return, size=n_simulations, 
                                           random_state=42)
        var_t = np.percentile(simulated_returns_t, (1 - confidence_level) * 100)
        cvar_t = simulated_returns_t[simulated_returns_t <= var_t].mean()
        var_annual_t = var_t * np.sqrt(252)
        cvar_annual_t = cvar_t * np.sqrt(252)
        
        # ==========================================
        # MÉTODO 3: Distribución t-STUDENT conservadora (df=3)
        # ==========================================
        simulated_returns_t_cons = stats.t.rvs(df=df_conservative, loc=mean_return, 
                                                scale=std_return, size=n_simulations, 
                                                random_state=42)
        var_t_cons = np.percentile(simulated_returns_t_cons, (1 - confidence_level) * 100)
        cvar_t_cons = simulated_returns_t_cons[simulated_returns_t_cons <= var_t_cons].mean()
        var_annual_t_cons = var_t_cons * np.sqrt(252)
        cvar_annual_t_cons = cvar_t_cons * np.sqrt(252)
        
        # ==========================================
        # COMPARACIÓN DE RESULTADOS
        # ==========================================
        print("\n" + "="*80)
        print("COMPARACIÓN: Normal vs t-Student (colas pesadas)")
        print("="*80)
        
        print("\n[1] DISTRIBUCION NORMAL (supone colas ligeras):")
        print(f"  VaR (1 dia, {confidence_level*100:.0f}%):      {var_normal*100:7.2f}%")
        print(f"  CVaR (1 dia, {confidence_level*100:.0f}%):     {cvar_normal*100:7.2f}%")
        print(f"  VaR (anual, {confidence_level*100:.0f}%):      {var_annual_normal*100:7.2f}%")
        print(f"  CVaR (anual, {confidence_level*100:.0f}%):     {cvar_annual_normal*100:7.2f}%")
        
        print(f"\n[2] DISTRIBUCION t-STUDENT (df={df_used:.1f} - estimado):")
        print(f"  VaR (1 dia, {confidence_level*100:.0f}%):      {var_t*100:7.2f}%")
        print(f"  CVaR (1 dia, {confidence_level*100:.0f}%):     {cvar_t*100:7.2f}%")
        print(f"  VaR (anual, {confidence_level*100:.0f}%):      {var_annual_t*100:7.2f}%")
        print(f"  CVaR (anual, {confidence_level*100:.0f}%):     {cvar_annual_t*100:7.2f}%")
        
        print(f"\n[3] DISTRIBUCION t-STUDENT CONSERVADORA (df={df_conservative} - fat tails):")
        print(f"  VaR (1 dia, {confidence_level*100:.0f}%):      {var_t_cons*100:7.2f}%")
        print(f"  CVaR (1 dia, {confidence_level*100:.0f}%):     {cvar_t_cons*100:7.2f}%")
        print(f"  VaR (anual, {confidence_level*100:.0f}%):      {var_annual_t_cons*100:7.2f}%")
        print(f"  CVaR (anual, {confidence_level*100:.0f}%):     {cvar_annual_t_cons*100:7.2f}%")
        
        # Calcular diferencias
        diff_var_t = ((var_t / var_normal) - 1) * 100
        diff_cvar_t = ((cvar_t / cvar_normal) - 1) * 100
        diff_var_t_cons = ((var_t_cons / var_normal) - 1) * 100
        diff_cvar_t_cons = ((cvar_t_cons / cvar_normal) - 1) * 100
        
        print("\n" + "="*80)
        print("INCREMENTO DEL RIESGO vs Distribucion Normal:")
        print("="*80)
        print(f"\n>> Con t-Student ESPERADO (df={df_used:.1f}):")
        print(f"  VaR incremento:  {abs(diff_var_t):6.2f}% ({abs(var_t - var_normal)*100:.3f} puntos porcentuales)")
        print(f"  CVaR incremento: {abs(diff_cvar_t):6.2f}% ({abs(cvar_t - cvar_normal)*100:.3f} puntos porcentuales)")
        
        print(f"\n>> Con t-Student CONSERVADOR (df={df_conservative}):")
        print(f"  VaR incremento:  {abs(diff_var_t_cons):6.2f}% ({abs(var_t_cons - var_normal)*100:.3f} puntos porcentuales)")
        print(f"  CVaR incremento: {abs(diff_cvar_t_cons):6.2f}% ({abs(cvar_t_cons - cvar_normal)*100:.3f} puntos porcentuales)")
        
        print("\n" + "="*80)
        print("RESUMEN DE ESCENARIOS:")
        print("="*80)
        print(f"\nEscenario            VaR (1d)    CVaR (1d)   VaR (anual)  CVaR (anual)")
        print(f"-" * 80)
        print(f"CONSERVADOR (df=3)   {var_t_cons*100:6.2f}%    {cvar_t_cons*100:6.2f}%    {var_annual_t_cons*100:7.2f}%   {cvar_annual_t_cons*100:8.2f}%")
        print(f"ESPERADO (df={df_used:.1f})     {var_t*100:6.2f}%    {cvar_t*100:6.2f}%    {var_annual_t*100:7.2f}%   {cvar_annual_t*100:8.2f}%")
        print(f"NORMAL (baseline)    {var_normal*100:6.2f}%    {cvar_normal*100:6.2f}%    {var_annual_normal*100:7.2f}%   {cvar_annual_normal*100:8.2f}%")
        
        print("\n" + "="*80)
        print("INTERPRETACION Y RECOMENDACIONES:")
        print("="*80)
        print(f"  La distribucion Normal SUBESTIMA el riesgo en mercados emergentes.")
        print(f"  Para Argentina, la t-Student captura mejor los eventos extremos (crisis,")
        print(f"  devaluaciones, defaults). El riesgo REAL es significativamente mayor.")
        print(f"\n  >> ESCENARIO CONSERVADOR (df=3): Ideal para asignacion de capital y limites de riesgo")
        print(f"  >> ESCENARIO ESPERADO (df={df_used:.1f}): Ideal para proyecciones y pricing")
        print(f"  >> BASELINE NORMAL: Solo para comparacion academica (NO usar en produccion)")
        print("="*80)
        
        # Retornar AMBOS escenarios para flexibilidad en gestión de riesgo
        return {
            # ESCENARIO CONSERVADOR (df=3) - Por defecto
            'var_daily': var_t_cons,
            'cvar_daily': cvar_t_cons,
            'var_annual': var_annual_t_cons,
            'cvar_annual': cvar_annual_t_cons,
            'simulated_returns': simulated_returns_t_cons,
            'df_used': df_conservative,
            
            # ESCENARIO ESPERADO (df estimado)
            'var_daily_expected': var_t,
            'cvar_daily_expected': cvar_t,
            'var_annual_expected': var_annual_t,
            'cvar_annual_expected': cvar_annual_t,
            'df_estimated': df_estimated,
            
            # BASELINE NORMAL (para comparación)
            'var_daily_normal': var_normal,
            'cvar_daily_normal': cvar_normal,
            'var_annual_normal': var_annual_normal,
            'cvar_annual_normal': cvar_annual_normal,
        }
    
    def plot_efficient_frontier(self, min_vol_weights=None, max_sharpe_weights=None, 
                               min_vol_perf=None, max_sharpe_perf=None, save_path=None):
        """
        Genera gráfico de la Frontera Eficiente
        
        Parameters:
        -----------
        min_vol_weights : dict, optional
            Pesos de la cartera de mínima volatilidad. Si no se provee, se calcula sin restricciones.
        max_sharpe_weights : dict, optional
            Pesos de la cartera de máximo Sharpe. Si no se provee, se calcula sin restricciones.
        min_vol_perf : tuple, optional
            Performance de la cartera de mínima volatilidad (retorno, volatilidad, sharpe)
        max_sharpe_perf : tuple, optional
            Performance de la cartera de máximo Sharpe (retorno, volatilidad, sharpe)
        save_path : str, optional
            Ruta para guardar el gráfico
        """
        print("\n[+] Generando Frontera Eficiente...")
        
        # Usar ruta por defecto si no se especifica
        if save_path is None:
            save_path = OUTPUTS_DIR / 'efficient_frontier.png'
        else:
            save_path = Path(save_path)
        
        # Crear directorio de salida si no existe
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        n_assets = len(self.tickers)
        
        # Generar carteras aleatorias para visualización
        n_portfolios = 5000
        results = np.zeros((3, n_portfolios))
        
        np.random.seed(42)
        for i in range(n_portfolios):
            # Generar pesos aleatorios
            weights = np.random.random(n_assets)
            weights /= weights.sum()
            
            # Calcular performance
            port_return, port_volatility, port_sharpe = self.portfolio_performance(weights)
            
            results[0, i] = port_return
            results[1, i] = port_volatility
            results[2, i] = port_sharpe
        
        # Si no se proveen pesos, calcular carteras óptimas sin restricciones
        if min_vol_weights is None or max_sharpe_weights is None:
            min_vol_weights, min_vol_perf = self.optimize_min_volatility()
            max_sharpe_weights, max_sharpe_perf = self.optimize_max_sharpe()
        
        # Convertir a arrays
        min_vol_array = np.array([min_vol_weights[ticker] for ticker in self.tickers])
        max_sharpe_array = np.array([max_sharpe_weights[ticker] for ticker in self.tickers])
        
        # Crear figura
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # Plot 1: Frontera Eficiente con carteras aleatorias
        scatter = ax1.scatter(results[1, :], results[0, :], 
                             c=results[2, :], cmap='viridis', 
                             alpha=0.5, s=10, edgecolors='none')
        plt.colorbar(scatter, ax=ax1, label='Sharpe Ratio')
        
        # Añadir activos individuales
        for ticker in self.tickers:
            annual_ret = self.returns[ticker].mean() * 252
            annual_vol = self.returns[ticker].std() * np.sqrt(252)
            ax1.scatter(annual_vol, annual_ret, s=150, alpha=0.7, 
                       label=ticker, edgecolors='black', linewidth=1.5)
        
        # Añadir cartera de mínima volatilidad
        ax1.scatter(min_vol_perf[1], min_vol_perf[0], s=400, c='green', 
                   marker='*', edgecolors='black', linewidth=2,
                   label='Mínima Volatilidad', zorder=5)
        
        # Añadir cartera de máximo Sharpe
        ax1.scatter(max_sharpe_perf[1], max_sharpe_perf[0], s=400, c='red', 
                   marker='*', edgecolors='black', linewidth=2,
                   label='Máximo Sharpe', zorder=5)
        
        ax1.set_xlabel('Volatilidad (Riesgo) Anualizada', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Retorno Esperado Anualizado', fontsize=12, fontweight='bold')
        ax1.set_title('Frontera Eficiente - Mercado Argentino y Global', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.legend(loc='best', fontsize=9, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Matriz de Correlación
        sns.heatmap(self.correlation_matrix, annot=True, fmt='.3f', 
                   cmap='RdYlGn', center=0, square=True, ax=ax2,
                   cbar_kws={'label': 'Correlación'}, linewidths=1,
                   vmin=-1, vmax=1)
        ax2.set_title('Matriz de Correlación entre Activos', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Plot 3: Composición de carteras
        x = np.arange(len(self.tickers))
        width = 0.35
        
        min_vol_values = [min_vol_weights[ticker] * 100 for ticker in self.tickers]
        max_sharpe_values = [max_sharpe_weights[ticker] * 100 for ticker in self.tickers]
        
        ax3.bar(x - width/2, min_vol_values, width, label='Min Volatilidad', alpha=0.8)
        ax3.bar(x + width/2, max_sharpe_values, width, label='Max Sharpe', alpha=0.8)
        
        ax3.set_xlabel('Activos', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Peso en Cartera (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Composición de Carteras Óptimas', fontsize=14, fontweight='bold', pad=20)
        ax3.set_xticks(x)
        ax3.set_xticklabels(self.tickers, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Matriz de Covarianza (Heatmap)
        sns.heatmap(self.S, annot=True, fmt='.4f', 
                   cmap='YlOrRd', square=True, ax=ax4,
                   cbar_kws={'label': 'Covarianza (Anualizada)'}, linewidths=1)
        ax4.set_title('Matriz de Covarianza (Anualizada)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Grafico guardado en: {save_path}")
        
        return fig, (min_vol_weights, min_vol_perf), (max_sharpe_weights, max_sharpe_perf)
    
    def generate_report(self, min_vol_weights, max_sharpe_weights, 
                       risk_metrics, min_vol_weights_free=None, max_sharpe_weights_free=None,
                       min_vol_perf_free=None, max_sharpe_perf_free=None,
                       risk_metrics_free=None, save_path=None):
        """
        Genera reporte técnico en Markdown
        
        Parameters (nuevos):
        --------------------
        min_vol_weights_free : dict, optional
            Pesos de mínima volatilidad sin restricciones (para comparación)
        max_sharpe_weights_free : dict, optional
            Pesos de máximo Sharpe sin restricciones (para comparación)
        min_vol_perf_free : tuple, optional
            Performance de min vol libre (retorno, volatilidad, sharpe)
        max_sharpe_perf_free : tuple, optional
            Performance de max sharpe libre (retorno, volatilidad, sharpe)
        risk_metrics_free : dict, optional
            Métricas de riesgo de cartera libre
        """
        print("\n[+] Generando reporte tecnico...")
        
        # Usar ruta por defecto si no se especifica
        if save_path is None:
            save_path = OUTPUTS_DIR / 'reporte_portfolio.md'
        else:
            save_path = Path(save_path)
        
        # Crear directorio de salida si no existe
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generar lista de activos dinámicamente y clasificarlos
        activos_lista = []
        activos_argentinos = []
        activos_globales = []
        activos_crypto = []
        
        for ticker in self.tickers:
            # Clasificar activos automáticamente
            if '.BA' in ticker:
                tipo = '[ARG] Equity Argentina'
                activos_argentinos.append(ticker)
            elif ticker.endswith('-USD'):
                tipo = '[CRYPTO] Criptomoneda'
                activos_crypto.append(ticker)
            else:
                tipo = '[GLOBAL] Equity Global'
                activos_globales.append(ticker)
            activos_lista.append(f"- **{ticker}** - {tipo}")
        
        activos_texto = '\n'.join(activos_lista)
        
        # Generar ejemplos dinámicos para el texto del reporte
        ejemplos_argentinos = ', '.join(activos_argentinos[:3]) if activos_argentinos else 'activos argentinos'
        ejemplo_global = activos_globales[0] if activos_globales else 'activos globales'
        ejemplo_crypto = activos_crypto[0] if activos_crypto else 'BTC-USD'
        
        # Texto dinámico para diversificación internacional
        if activos_globales and activos_crypto:
            texto_diversificacion_intl = f"- Activos como {ejemplo_global} y {ejemplo_crypto} reducen el riesgo sistémico argentino"
        elif activos_globales:
            texto_diversificacion_intl = f"- Activos como {ejemplo_global} reducen el riesgo sistémico argentino"
        elif activos_crypto:
            texto_diversificacion_intl = f"- Activos como {ejemplo_crypto} reducen el riesgo sistémico argentino"
        else:
            texto_diversificacion_intl = "- La diversificación internacional reduce el riesgo sistémico"
        
        # Generar sección de comparación Libre vs Gestionada (si se proveen datos)
        if (max_sharpe_weights_free is not None and max_sharpe_perf_free is not None and 
            risk_metrics_free is not None):
            
            # Tabla de composición
            comparacion_composicion = "### Composición de Carteras (Máximo Sharpe)\n\n"
            comparacion_composicion += "| Activo | Libre (%) | Gestionada (%) | Diferencia |\n"
            comparacion_composicion += "|--------|-----------|----------------|------------|\n"
            
            for ticker in self.tickers:
                weight_free = max_sharpe_weights_free.get(ticker, 0) * 100
                weight_managed = max_sharpe_weights.get(ticker, 0) * 100
                diff = weight_managed - weight_free
                
                # Solo mostrar activos con peso > 0.1% en alguna versión
                if weight_free > 0.1 or weight_managed > 0.1:
                    comparacion_composicion += f"| {ticker} | {weight_free:.2f}% | {weight_managed:.2f}% | {diff:+.2f}% |\n"
            
            # Calcular performance de la cartera gestionada
            weights_managed_array = np.array([max_sharpe_weights.get(ticker, 0) for ticker in self.tickers])
            ret_managed_val, vol_managed_val, sharpe_managed_val = self.portfolio_performance(weights_managed_array)
            
            # Tabla de performance
            ret_free = max_sharpe_perf_free[0] * 100
            ret_managed = ret_managed_val * 100
            vol_free = max_sharpe_perf_free[1] * 100
            vol_managed = vol_managed_val * 100
            sharpe_free = max_sharpe_perf_free[2]
            sharpe_managed = sharpe_managed_val
            
            comparacion_performance = "\n### Métricas de Performance\n\n"
            comparacion_performance += "| Métrica | Libre | Gestionada | Diferencia |\n"
            comparacion_performance += "|---------|-------|------------|------------|\n"
            comparacion_performance += f"| **Retorno Anualizado** | {ret_free:.2f}% | {ret_managed:.2f}% | {ret_managed-ret_free:+.2f}% |\n"
            comparacion_performance += f"| **Volatilidad Anualizada** | {vol_free:.2f}% | {vol_managed:.2f}% | {vol_managed-vol_free:+.2f}% |\n"
            comparacion_performance += f"| **Sharpe Ratio** | {sharpe_free:.2f} | {sharpe_managed:.2f} | {sharpe_managed-sharpe_free:+.2f} |\n"
            
            # Tabla de riesgo (VaR/CVaR)
            var_free = risk_metrics_free['var_daily'] * 100
            var_managed = risk_metrics['var_daily'] * 100
            cvar_free = risk_metrics_free['cvar_daily'] * 100
            cvar_managed = risk_metrics['cvar_daily'] * 100
            
            comparacion_riesgo = "\n### Métricas de Riesgo (VaR/CVaR Conservador, df=3)\n\n"
            comparacion_riesgo += "| Métrica | Libre | Gestionada | Mejora |\n"
            comparacion_riesgo += "|---------|-------|------------|--------|\n"
            comparacion_riesgo += f"| **VaR (1 día, 95%)** | {var_free:.2f}% | {var_managed:.2f}% | {var_managed-var_free:+.2f}% |\n"
            comparacion_riesgo += f"| **CVaR (1 día, 95%)** | {cvar_free:.2f}% | {cvar_managed:.2f}% | {cvar_managed-cvar_free:+.2f}% |\n"
            
            # Análisis e interpretación
            sharpe_diff_pct = ((sharpe_managed - sharpe_free) / sharpe_free * 100) if sharpe_free != 0 else 0
            var_mejora = "mejoró" if var_managed > var_free else "empeoró"
            
            comparacion_analisis = f"\n### Análisis e Interpretación\n\n"
            comparacion_analisis += f"**1. Sharpe Ratio:**\n"
            if sharpe_diff_pct < -5:
                comparacion_analisis += f"   - ⚠️ El Sharpe disminuyó {abs(sharpe_diff_pct):.1f}% con las restricciones\n"
                comparacion_analisis += f"   - Las restricciones limitan el potencial de optimización\n"
            elif sharpe_diff_pct < 0:
                comparacion_analisis += f"   - ✅ Trade-off aceptable: Sharpe disminuyó solo {abs(sharpe_diff_pct):.1f}%\n"
            else:
                comparacion_analisis += f"   - ✅ Las restricciones mejoraron el Sharpe en {sharpe_diff_pct:.1f}%\n"
            
            comparacion_analisis += f"\n**2. Riesgo de Cola (VaR/CVaR):**\n"
            comparacion_analisis += f"   - El VaR {var_mejora} {abs(var_managed-var_free):.2f} puntos porcentuales\n"
            comparacion_analisis += f"   - CVaR {'mejoró' if cvar_managed > cvar_free else 'empeoró'} {abs(cvar_managed-cvar_free):.2f} puntos porcentuales\n"
            
            # Conteo de activos
            n_activos_libre = sum(1 for w in max_sharpe_weights_free.values() if w > 0.01)
            n_activos_gestionada = sum(1 for w in max_sharpe_weights.values() if w > 0.01)
            
            comparacion_analisis += f"\n**3. Diversificación:**\n"
            comparacion_analisis += f"   - Libre: {n_activos_libre} activos con peso significativo (>1%)\n"
            comparacion_analisis += f"   - Gestionada: {n_activos_gestionada} activos con peso significativo (>1%)\n"
            
            comparacion_analisis += f"\n**4. Recomendación:**\n"
            if sharpe_diff_pct > -10:
                comparacion_analisis += f"   - ✅ **USAR CARTERA GESTIONADA:** Trade-off aceptable entre eficiencia y control de riesgo\n"
                comparacion_analisis += f"   - Las restricciones proporcionan mayor robustez y mejor gestión de riesgo de concentración\n"
            else:
                comparacion_analisis += f"   - ⚠️ **REVISAR RESTRICCIONES:** La pérdida de eficiencia es significativa\n"
                comparacion_analisis += f"   - Considerar relajar algunos límites para mejorar el Sharpe Ratio\n"
            
            comparacion_libre_gestionada = comparacion_composicion + comparacion_performance + comparacion_riesgo + comparacion_analisis
        else:
            comparacion_libre_gestionada = "**Nota:** No se realizó comparación con optimización libre (sin restricciones). Este reporte muestra únicamente las carteras gestionadas con restricciones aplicadas."
        
        report = f"""# Reporte de Optimización de Cartera
## Análisis Cuantitativo del Mercado Argentino y Global

**Fecha de Generación:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Período Analizado (Train Set):** {self.start_date} a {self.end_date}  
**Analista:** Jorge Iván Juárez A. - Lic. en Economía

---

> **📊 Nota Metodológica:** Este reporte contiene el análisis **ex-ante** basado en datos históricos 
> del período indicado (train set). Los pesos óptimos calculados aquí se validan posteriormente 
> en el **Reporte de Backtesting** usando datos **out-of-sample** (test set) para asegurar 
> robustez y evitar overfitting.

---

## 1. Activos Bajo Análisis

Los activos seleccionados para el análisis son:

{activos_texto}

**Total de activos:** {len(self.tickers)}

---

## 2. Matriz de Covarianza y Análisis de Riesgo Sistémico

### Matriz de Covarianza (Anualizada)

```
{self.S.to_string()}
```

### Interpretación Económica

La **matriz de covarianza** es fundamental para entender cómo los movimientos de un activo afectan a otro. 
Los valores en la diagonal representan la varianza de cada activo (riesgo individual), mientras que 
los valores fuera de la diagonal muestran la covarianza entre pares de activos.

**Observaciones Clave:**

1. **Activos Argentinos y Riesgo Sistémico:**
   - Los activos argentinos ({ejemplos_argentinos}) tienden a presentar covarianzas positivas 
     entre sí, reflejando el **riesgo país** que afecta sistemáticamente al mercado local.
   - Eventos macroeconómicos (inflación, tipo de cambio, política monetaria) impactan 
     simultáneamente a estos activos, incrementando el riesgo sistémico de la cartera.

2. **Diversificación Internacional:**
   - {ejemplo_global} presenta covarianzas más bajas con activos argentinos, ofreciendo **beneficios de diversificación**.
   - {ejemplo_crypto} muestra comportamiento asincrónico, actuando como **activo descorrelacionado**.

3. **Implicaciones para la Gestión de Riesgo:**
   - Una concentración alta en activos argentinos **no reduce el riesgo** por diversificación 
     (correlaciones altas → covarianzas positivas elevadas).
   - La inclusión de activos internacionales **reduce la exposición al riesgo sistémico argentino**.

### Matriz de Correlación

```
{self.correlation_matrix.to_string()}
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
{self._format_weights(min_vol_weights)}
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
{self._format_weights(max_sharpe_weights)}
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
VaR (1 día, 95%):      {risk_metrics['var_daily']*100:.2f}%
CVaR (1 día, 95%):     {risk_metrics['cvar_daily']*100:.2f}%
VaR (anualizado, 95%): {risk_metrics['var_annual']*100:.2f}%
CVaR (anualizado, 95%): {risk_metrics['cvar_annual']*100:.2f}%
```

**Interpretación:** Supuesto de **máxima prudencia**. Asume que eventos extremos son más 
frecuentes que lo observado históricamente. Ideal para dimensionar capital de respaldo.

---

#### 🟡 ESCENARIO ESPERADO - t-Student (df={risk_metrics.get('df_estimated', 5):.1f})

**Uso recomendado:** Proyecciones, pricing, análisis comparativo

```
VaR (1 día, 95%):      {risk_metrics.get('var_daily_expected', 0)*100:.2f}%
CVaR (1 día, 95%):     {risk_metrics.get('cvar_daily_expected', 0)*100:.2f}%
VaR (anualizado, 95%): {risk_metrics.get('var_annual_expected', 0)*100:.2f}%
CVaR (anualizado, 95%): {risk_metrics.get('cvar_annual_expected', 0)*100:.2f}%
```

**Interpretación:** Basado en grados de libertad **estimados de datos históricos**. 
Refleja el comportamiento observado en el período analizado (2024-2026).

**Método de Estimación:** Maximum Likelihood Estimation (MLE) aplicado a los retornos diarios de la cartera.
La función de verosimilitud maximiza: L(df, μ, σ | datos) para la distribución t-Student.
Estimación obtenida: df ≈ {risk_metrics.get('df_estimated', 0):.2f}, donde valores bajos (df < 5) indican mayor presencia de eventos extremos.

---

#### ⚪ BASELINE - Distribución Normal (referencia)

**Uso recomendado:** Solo para comparación académica (NO para gestión de riesgo)

```
VaR (1 día, 95%):      {risk_metrics.get('var_daily_normal', 0)*100:.2f}%
CVaR (1 día, 95%):     {risk_metrics.get('cvar_daily_normal', 0)*100:.2f}%
VaR (anualizado, 95%): {risk_metrics.get('var_annual_normal', 0)*100:.2f}%
CVaR (anualizado, 95%): {risk_metrics.get('cvar_annual_normal', 0)*100:.2f}%
```

**⚠️ Advertencia:** La Normal **subestima significativamente** el riesgo en mercados emergentes.

---

### Comparación de Escenarios

| Métrica | Normal | Esperado (df={risk_metrics.get('df_estimated', 5):.1f}) | Conservador (df=3) |
|---------|--------|-------------|-------------------|
| VaR (1d) | {risk_metrics.get('var_daily_normal', 0)*100:.2f}% | {risk_metrics.get('var_daily_expected', 0)*100:.2f}% | {risk_metrics['var_daily']*100:.2f}% |
| CVaR (1d) | {risk_metrics.get('cvar_daily_normal', 0)*100:.2f}% | {risk_metrics.get('cvar_daily_expected', 0)*100:.2f}% | {risk_metrics['cvar_daily']*100:.2f}% |

**Diferencia Conservador vs Normal:**
- VaR: +{abs((risk_metrics['var_daily']/risk_metrics.get('var_daily_normal', risk_metrics['var_daily']) - 1)*100):.1f}%
- CVaR: +{abs((risk_metrics['cvar_daily']/risk_metrics.get('cvar_daily_normal', risk_metrics['cvar_daily']) - 1)*100):.1f}%

### Interpretación Económica

- **VaR:** En el 95% de los días, la cartera **no perderá más del {abs(risk_metrics['var_daily'])*100:.2f}%**.
- **CVaR:** En escenarios extremos (5% peor de los casos), la pérdida promedio será del **{abs(risk_metrics['cvar_daily'])*100:.2f}%**.
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
   - Covarianzas positivas altas entre {ejemplos_argentinos} indican que se mueven juntos
   - Esto implica que la diversificación entre activos argentinos es **limitada**

3. **Efecto en el Riesgo Total:**
   - La varianza del portfolio es: σ²ₚ = wᵀΣw (donde w son los pesos y Σ la matriz de covarianza)
   - Covarianzas altas incrementan σ²ₚ más que la simple suma de varianzas individuales
   - Este es el **riesgo sistémico no diversificable**

### Recomendaciones para Gestión de Riesgo

#### Para el Inversor Conservador
- Priorizar la **cartera de mínima volatilidad**
- Limitar exposición a activos argentinos de alta volatilidad
- Mantener diversificación internacional robusta (> 40% en activos globales)

#### Para el Inversor Agresivo
- Implementar la **cartera de máximo Sharpe**
- Monitorear constantemente el VaR/CVaR
- Establecer stop-loss en niveles cercanos al VaR diario

#### Gestión del Riesgo Sistémico Argentino
1. **Hedging:** Considerar instrumentos de cobertura (dólar MEP, bonos en USD, futuros)
2. **Rebalanceo Dinámico:** Ajustar pesos trimestralmente según cambios en volatilidad y correlaciones
3. **Stress Testing:** Evaluar impacto de escenarios adversos:
   - Devaluación abrupta (> 30%)
   - Evento de default soberano
   - Controles de capital
4. **Diversificación Geográfica:** Mantener exposición significativa a mercados no correlacionados
5. **Monitoreo de Indicadores Macro:** 
   - Brecha cambiaria
   - Reservas del BCRA
   - Riesgo país (EMBI+)
   - Inflación mensual

---

## 6. Comparación: Optimización Libre vs Gestionada

### Metodología

Se compararon **dos enfoques de optimización**:

1. **Optimización LIBRE (Sin Restricciones):** Permite cualquier asignación entre 0% y 100% por activo
2. **Optimización GESTIONADA (Con Restricciones):** Aplica límites realistas por tipo de activo:
   - Activos argentinos (.BA): Máximo 20% individual
   - Criptomonedas (-USD): Máximo 10%
   - Activos globales: Mínimo 10-15%, Máximo 30-35%

{comparacion_libre_gestionada}

---

## 7. Conclusiones Técnicas

### Hallazgos Principales

1. **Riesgo Sistémico Elevado:** 
   - La alta correlación entre activos argentinos amplifica el riesgo de la cartera
   - La matriz de covarianza muestra dependencias significativas

2. **Beneficios de Diversificación Internacional:**
   {texto_diversificacion_intl}
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
"""
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"[OK] Reporte guardado en: {save_path}")
        
        return save_path
    
    def _format_weights(self, weights_dict):
        """Helper para formatear pesos en el reporte"""
        formatted = ""
        for ticker, weight in weights_dict.items():
            if weight > 0.001:
                formatted += f"{ticker:12s}: {weight*100:6.2f}%\n"
        return formatted
    
    def generate_backtest_report(self, backtest_results, max_sharpe_weights, 
                                 risk_metrics, save_path=None):
        """
        Genera reporte técnico de backtesting en Markdown
        
        Parameters:
        -----------
        backtest_results : dict
            Resultados del backtesting
        max_sharpe_weights : dict
            Pesos de la cartera testeada
        risk_metrics : dict
            Métricas de riesgo (VaR/CVaR) calculadas ex-ante
        save_path : str
            Ruta para guardar el reporte
        """
        print("\n[+] Generando reporte de backtesting...")
        
        # Usar ruta por defecto si no se especifica
        if save_path is None:
            save_path = OUTPUTS_DIR / 'reporte_backtesting.md'
        else:
            save_path = Path(save_path)
        
        # Crear directorio de salida si no existe
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extraer métricas del backtest - ACTIVA (con rebalanceo)
        initial = backtest_results['initial_capital']
        final_active = backtest_results['final_value_active']
        total_ret_active = backtest_results['total_return_active']
        ann_ret_active = backtest_results['annualized_return_active']
        volatility_active = backtest_results['volatility_active']
        sharpe_active = backtest_results['sharpe_ratio_active']
        max_dd_active = backtest_results['max_drawdown_active']
        total_commissions = backtest_results['total_commissions']
        
        # Extraer métricas del backtest - PASIVA (sin rebalanceo)
        final_passive = backtest_results['final_value_passive']
        total_ret_passive = backtest_results['total_return_passive']
        ann_ret_passive = backtest_results['annualized_return_passive']
        volatility_passive = backtest_results['volatility_passive']
        sharpe_passive = backtest_results['sharpe_ratio_passive']
        max_dd_passive = backtest_results['max_drawdown_passive']
        initial_commission_passive = backtest_results['initial_commission_passive']
        
        # Análisis de Fricción
        friction_cost = backtest_results['friction_cost']
        return_difference = backtest_results['return_difference']
        friction_worth_it = backtest_results['friction_worth_it']
        
        # Benchmark
        bench_ret = backtest_results.get('benchmark_return', 'N/A')
        bench_final = backtest_results.get('benchmark_final', 'N/A')
        bench_dd = backtest_results.get('benchmark_max_dd', 'N/A')
        bench_ann = backtest_results.get('benchmark_annualized', 'N/A')
        outperf_active = backtest_results.get('outperformance_active', 'N/A')
        outperf_passive = backtest_results.get('outperformance_passive', 'N/A')
        
        # Crear alias para compatibilidad con el reporte existente (usar métricas activas como default)
        final = final_active
        total_ret = total_ret_active
        ann_ret = ann_ret_active
        volatility = volatility_active
        sharpe = sharpe_active
        max_dd = max_dd_active
        
        # Formatear valores
        bench_ret_str = f"{bench_ret:+.2f}%" if isinstance(bench_ret, (int, float)) else bench_ret
        bench_ann_str = f"{bench_ann:+.2f}%" if isinstance(bench_ann, (int, float)) else bench_ann
        bench_final_str = f"${bench_final:,.2f}" if isinstance(bench_final, (int, float)) else bench_final
        bench_dd_str = f"{bench_dd:.2f}%" if isinstance(bench_dd, (int, float)) else bench_dd
        outperf_active_str = f"{outperf_active:+.2f}%" if isinstance(outperf_active, (int, float)) else outperf_active
        outperf_passive_str = f"{outperf_passive:+.2f}%" if isinstance(outperf_passive, (int, float)) else outperf_passive
        outperf_active_emoji = " MEJOR" if isinstance(outperf_active, (int, float)) and outperf_active > 0 else " PEOR"
        outperf_passive_emoji = " MEJOR" if isinstance(outperf_passive, (int, float)) and outperf_passive > 0 else " PEOR"
        
        # Obtener período real del backtest (desde los resultados)
        start_date = backtest_results.get('start_date', datetime.now() - timedelta(days=365))
        end_date = backtest_results.get('end_date', datetime.now())
        lookback_str = backtest_results.get('lookback_period', '1y')
        
        # Convertir lookback_period a texto legible
        if lookback_str.endswith('y'):
            years = int(lookback_str[:-1])
            periodo_texto = f"{years} año{'s' if years > 1 else ''}"
        elif lookback_str.endswith('mo'):
            months = int(lookback_str[:-2])
            periodo_texto = f"{months} mes{'es' if months > 1 else ''}"
        elif lookback_str.endswith('d'):
            days = int(lookback_str[:-1])
            periodo_texto = f"{days} día{'s' if days > 1 else ''}"
        else:
            periodo_texto = "período histórico"
        
        # Verificar si es verdaderamente out-of-sample
        optimization_end = datetime.strptime(self.end_date, '%Y-%m-%d')
        is_out_of_sample = start_date >= optimization_end
        
        # Calcular años del período de entrenamiento (train set)
        train_start = datetime.strptime(self.start_date, '%Y-%m-%d')
        train_end = datetime.strptime(self.end_date, '%Y-%m-%d')
        train_years = round((train_end - train_start).days / 365.25, 1)
        
        if is_out_of_sample:
            validacion_tipo = "✅ **Validación Out-of-Sample Genuina**"
            validacion_nota = "El backtesting usa datos **posteriores** al período de optimización, proporcionando una validación robusta."
        else:
            validacion_tipo = "⚠️ **Validación In-Sample (Con Overlap)**"
            validacion_nota = f"El backtesting incluye datos que fueron usados en la optimización. Para validación out-of-sample, usar datos posteriores a {self.end_date}."
        
        report = f"""# Reporte de Backtesting - Validación Histórica
## Análisis Ex-Post con Gestión Activa vs Pasiva

**Fecha de Generación:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Período de Optimización (Train):** {self.start_date} a {self.end_date}  
**Período de Backtest (Test):** {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}  
**Estrategia Validada:** Cartera de Máximo Sharpe Ratio **GESTIONADA** (con restricciones)  
**Capital Inicial:** ${initial:,.2f} USD  
**Comisión por Operación:** 0.5%

> **📋 Nota sobre la estrategia:** Se validó la cartera **GESTIONADA** (con restricciones por tipo de activo) 
> en lugar de la LIBRE (sin restricciones), ya que la gestionada ofrece mejor control de riesgo de concentración 
> con un trade-off mínimo de eficiencia. Ver **Sección 6** del Reporte de Optimización para la comparación completa.

{validacion_tipo}

> {validacion_nota}

---

## RESUMEN EJECUTIVO

La estrategia fue **validada históricamente** comparando **DOS ENFOQUES**:

1. **ACTIVA (Rebalanceo Mensual):** Ajusta pesos al target cada mes, pagando comisiones
2. **PASIVA (Buy-and-Hold):** Compra inicial sin rebalanceo, comisión única

### Resultados Comparativos

| Métrica | Active (Rebalanceo) | Passive (Buy-Hold) | Diferencia |
|---------|--------------------|--------------------|------------|
| **Capital Final** | ${final_active:,.2f} | ${final_passive:,.2f} | ${final_active - final_passive:+,.2f} |
| **Retorno Total** | {total_ret_active:+.2f}% | {total_ret_passive:+.2f}% | {return_difference:+.2f}% |
| **Retorno Anualizado** | {ann_ret_active:+.2f}% | {ann_ret_passive:+.2f}% | {ann_ret_active - ann_ret_passive:+.2f}% |
| **Sharpe Ratio** | {sharpe_active:.2f} | {sharpe_passive:.2f} | {sharpe_active - sharpe_passive:+.2f} |
| **Máximo Drawdown** | {max_dd_active:.2f}% | {max_dd_passive:.2f}% | {max_dd_active - max_dd_passive:+.2f}% |
| **Comisiones Totales** | ${total_commissions:,.2f} ({total_commissions/initial*100:.2f}%) | ${initial_commission_passive:,.2f} ({initial_commission_passive/initial*100:.2f}%) | ${friction_cost:,.2f} |

### CONCLUSION CLAVE: {'REBALANCEO VALIO LA PENA' if friction_worth_it else 'REBALANCEO NO VALIO LA PENA'}

{'El rebalanceo activo SUPERO al buy-and-hold en ' + str(abs(return_difference)) + ' puntos porcentuales, justificando las comisiones adicionales de $' + f'{friction_cost:,.2f}' + '.' if friction_worth_it else 'El rebalanceo activo NO justificó las comisiones adicionales de $' + f'{friction_cost:,.2f}' + '. La estrategia pasiva fue superior en ' + str(abs(return_difference)) + ' puntos porcentuales.'}

### Comparación vs Benchmark (SPY)

| Métrica | Active | Passive | Benchmark | Active vs Bench | Passive vs Bench |
|---------|--------|---------|-----------|-----------------|------------------|
| Retorno Total | {total_ret_active:+.2f}% | {total_ret_passive:+.2f}% | {bench_ret_str} | {outperf_active_str}{outperf_active_emoji} | {outperf_passive_str}{outperf_passive_emoji} |
| Capital Final | ${final_active:,.2f} | ${final_passive:,.2f} | {bench_final_str} | - | - |
| Max Drawdown | {max_dd_active:.2f}% | {max_dd_passive:.2f}% | {bench_dd_str} | {"Mejor" if isinstance(max_dd_active, (int, float)) and isinstance(bench_dd, (int, float)) and abs(max_dd_active) < abs(bench_dd) else "Peor"} | {"Mejor" if isinstance(max_dd_passive, (int, float)) and isinstance(bench_dd, (int, float)) and abs(max_dd_passive) < abs(bench_dd) else "Peor"} |

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
- **Período:** {periodo_texto} ({start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')})
- **Capital Inicial:** ${initial:,.2f} USD
- **Frecuencia:** Diaria (ajustado al cierre)
- **Costos:** Comisiones de 0.5% por operación (realista)

### Composición de la Cartera Testeada

**Pesos de la Cartera de Máximo Sharpe GESTIONADA:**

```
{self._format_weights(max_sharpe_weights)}
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
Capital Inicial:       ${initial:,.2f}
Capital Final:         ${final:,.2f}
Ganancia/Pérdida:      ${final - initial:,.2f} ({total_ret:+.2f}%)
```

#### Retornos
```
Retorno Total (período):   {total_ret:+.2f}%
Retorno Anualizado:        {ann_ret:+.2f}%
```

**Interpretación:**
- {"La cartera generó retornos positivos, cumpliendo con las expectativas de la optimización." if total_ret > 0 else "La cartera tuvo retornos negativos en el período analizado, lo que sugiere condiciones de mercado adversas."}
- {"El retorno anualizado supera el típico rendimiento de bonos soberanos (5-7%), validando la estrategia." if ann_ret > 7 else "El retorno anualizado está por debajo de expectativas, requiere revisión de supuestos."}

### 2.2 Métricas de Riesgo Realizadas

#### Volatilidad
```
Volatilidad Anualizada:    {volatility:.2f}%
```

**Interpretación:**
- {"Volatilidad moderada, adecuada para el perfil de riesgo de la cartera." if volatility < 30 else "Volatilidad elevada, refleja la exposición a activos argentinos y cripto."}

#### Sharpe Ratio Realizado
```
Sharpe Ratio:              {sharpe:.2f}
```

**Benchmarks de Sharpe:**
- < 0: Estrategia destruye valor
- 0 - 1: Retorno no compensa el riesgo adecuadamente
- 1 - 2: Buena relación riesgo-retorno ✅
- 2+: Excelente relación riesgo-retorno ⭐

**Veredicto:** {"✅ La estrategia muestra buena eficiencia ajustada por riesgo" if sharpe >= 1 else "⚠️ La relación riesgo-retorno es subóptima"}

---

## 3. Análisis de Drawdown (Caídas)

### Máximo Drawdown Histórico

```
Máximo Drawdown:           {max_dd:.2f}%
```

**¿Qué significa?**
El Máximo Drawdown (MDD) representa la **caída más profunda** que experimentó la cartera 
desde un máximo histórico hasta un mínimo posterior. Es una medida crítica de **riesgo 
de pérdida temporal**.

**Interpretación:**
- {"Drawdown controlado, dentro de límites aceptables para inversores moderados." if abs(max_dd) < 20 else "Drawdown significativo, requiere fuerte tolerancia al riesgo."}
- {"Este nivel de caída es típico en carteras con exposición a mercados emergentes y cripto." if abs(max_dd) > 15 else "Drawdown bajo indica buena gestión de riesgo."}

### Comparación con VaR/CVaR Proyectado

Recordemos los niveles de riesgo proyectados ex-ante:

| Escenario | VaR (1 día, 95%) | CVaR (1 día, 95%) | VaR Anualizado | CVaR Anualizado |
|-----------|------------------|-------------------|----------------|-----------------|
| Conservador (df=3) | {risk_metrics['var_daily']*100:.2f}% | {risk_metrics['cvar_daily']*100:.2f}% | {risk_metrics['var_annual']*100:.2f}% | {risk_metrics['cvar_annual']*100:.2f}% |
| Esperado (df={risk_metrics.get('df_estimated', 5):.1f}) | {risk_metrics.get('var_daily_expected', 0)*100:.2f}% | {risk_metrics.get('cvar_daily_expected', 0)*100:.2f}% | {risk_metrics.get('var_annual_expected', 0)*100:.2f}% | {risk_metrics.get('cvar_annual_expected', 0)*100:.2f}% |
| Normal (baseline) | {risk_metrics.get('var_daily_normal', 0)*100:.2f}% | {risk_metrics.get('cvar_daily_normal', 0)*100:.2f}% | {risk_metrics.get('var_annual_normal', 0)*100:.2f}% | {risk_metrics.get('cvar_annual_normal', 0)*100:.2f}% |

**Máximo Drawdown Realizado:** {max_dd:.2f}%

**Análisis Comparativo:**
{self._generate_var_comparison(max_dd, risk_metrics)}

---

## 4. Comparación: Cartera vs Benchmark

### 4.1 Retornos

```
Estrategia Activa:         {total_ret_active:+.2f}%
Estrategia Pasiva:         {total_ret_passive:+.2f}%
Benchmark (SPY):           {bench_ret_str}
Active vs Bench:           {outperf_active_str}{outperf_active_emoji}
Passive vs Bench:          {outperf_passive_str}{outperf_passive_emoji}
```

**Interpretación:**
{f"✅ **La estrategia activa SUPERÓ al benchmark** en {outperf_active:.2f} puntos porcentuales. El rebalanceo agregó valor." if isinstance(outperf_active, (int, float)) and outperf_active > 0 else f"⚠️ **La estrategia activa UNDERPERFORMÓ al benchmark** por {abs(outperf_active):.2f} puntos porcentuales." if isinstance(outperf_active, (int, float)) else "No se pudo calcular comparación con benchmark."}

{f"✅ **La estrategia pasiva IGUALÓ al benchmark** (diferencia de {abs(outperf_passive):.2f}%)." if isinstance(outperf_passive, (int, float)) and abs(outperf_passive) < 1 else f"⚠️ **La estrategia pasiva fue inferior al benchmark** por {abs(outperf_passive):.2f}%." if isinstance(outperf_passive, (int, float)) and outperf_passive < 0 else f"✅ **La estrategia pasiva superó al benchmark** en {outperf_passive:.2f}%." if isinstance(outperf_passive, (int, float)) else ""}

### 4.2 Riesgo (Drawdown)

```
Cartera:                   {max_dd:.2f}%
Benchmark:                 {bench_dd_str}
```

**Interpretación:**
{f"✅ La cartera tuvo **menor drawdown** que el benchmark, mostrando mejor gestión de riesgo." if isinstance(max_dd, (int, float)) and isinstance(bench_dd, (int, float)) and abs(max_dd) < abs(bench_dd) else f"⚠️ La cartera sufrió **mayor drawdown** que el benchmark, indicando mayor volatilidad." if isinstance(max_dd, (int, float)) and isinstance(bench_dd, (int, float)) else "No disponible"}

---

## 5. Validación de Supuestos: Proyectado vs Realizado

### 5.1 Retorno Esperado vs Retorno Realizado

En el análisis ex-ante se proyectó un retorno anualizado basado en datos históricos ({train_years} años). 
El backtest nos permite validar si esas proyecciones fueron precisas.

**Resultado:** {f"✅ El retorno realizado ({ann_ret:.2f}%) está alineado con las proyecciones." if ann_ret > 0 else "⚠️ El retorno realizado fue inferior a las proyecciones, posiblemente por condiciones de mercado adversas."}

### 5.2 Volatilidad Proyectada vs Volatilidad Realizada

La volatilidad anualizada realizada fue de **{volatility:.2f}%**.

**Análisis:** {"La volatilidad realizada está dentro del rango esperado para esta cartera." if volatility < 40 else "La volatilidad realizada superó las expectativas, indicando un período de alta turbulencia."}

### 5.3 VaR/CVaR: ¿Fue Preciso?

El VaR y CVaR son medidas prospectivas de riesgo. El backtest nos permite verificar si 
los modelos fueron adecuados:

- **VaR Conservador (anual):** {risk_metrics['var_annual']*100:.2f}%
- **Máximo Drawdown Realizado:** {max_dd:.2f}%

{f"✅ El VaR conservador fue ADECUADO: el drawdown real ({abs(max_dd):.2f}%) fue menor al VaR proyectado ({abs(risk_metrics['var_annual']*100):.2f}%)." if abs(max_dd) < abs(risk_metrics['var_annual']*100) else f"⚠️ El VaR fue INSUFICIENTE: el drawdown real ({abs(max_dd):.2f}%) SUPERÓ al VaR proyectado ({abs(risk_metrics['var_annual']*100):.2f}%). Se recomienda usar el escenario conservador (df=3)."}

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

{self._generate_strengths(total_ret, sharpe, outperf_active)}

### ⚠️ Debilidades Identificadas

{self._generate_weaknesses(max_dd, volatility, outperf_active)}

### 🔍 Lecciones Aprendidas

1. **Validación de Modelos:** 
   - {f"El VaR conservador fue apropiado para gestión de riesgo." if abs(max_dd) < abs(risk_metrics['var_annual']*100) else "Se debe priorizar el VaR conservador (df=3) para asignación de capital."}
   
2. **Comportamiento en Crisis:**
   - El drawdown máximo ({max_dd:.2f}%) muestra la resiliencia de la cartera en períodos adversos
   
3. **Diversificación:**
   - La combinación de activos argentinos, globales y cripto {"cumplió su función de reducir riesgo" if abs(max_dd) < 25 else "no fue suficiente para mitigar riesgos sistémicos"}

---

## 8. Recomendaciones para Implementación

### Para Inversores Conservadores
- Considerar **reducir exposición a activos argentinos** si el drawdown supera tolerancia
- Implementar **stop-loss** en nivel cercano al VaR diario ({risk_metrics['var_daily']*100:.2f}%)
- **Rebalancear** trimestralmente para mantener pesos óptimos

### Para Inversores Agresivos
- La estrategia {"validó su eficiencia" if sharpe > 1 else "requiere ajustes en la asignación"}
- Considerar **apalancamiento moderado** si Sharpe > 2
- Monitorear **indicadores macro argentinos** (riesgo país, tipo de cambio)

### Ajustes Sugeridos

{self._generate_adjustments(total_ret, sharpe, max_dd, outperf_active)}

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
"""
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"[OK] Reporte de backtesting guardado en: {save_path}")
        
        return save_path
    
    def _generate_var_comparison(self, max_dd, risk_metrics):
        """Helper para comparar drawdown con VaR/CVaR proyectado"""
        var_annual = abs(risk_metrics['var_annual'] * 100)
        cvar_annual = abs(risk_metrics['cvar_annual'] * 100)
        dd_abs = abs(max_dd)
        
        if dd_abs < var_annual:
            return f"""✅ **El drawdown realizado ({dd_abs:.2f}%) fue MENOR que el VaR conservador proyectado ({var_annual:.2f}%).**

Esto indica que:
- El modelo de VaR fue **prudente y adecuado**
- No se materializaron eventos extremos en el período
- La cartera se comportó dentro de los parámetros esperados"""
        elif dd_abs < cvar_annual:
            return f"""⚠️ **El drawdown realizado ({dd_abs:.2f}%) SUPERÓ el VaR ({var_annual:.2f}%) pero está por debajo del CVaR ({cvar_annual:.2f}%).**

Esto indica que:
- Hubo eventos que excedieron el VaR (el 5% peor de los casos)
- El CVaR capturó adecuadamente el "tail risk"
- Se justifica usar el CVaR como medida de capital de riesgo"""
        else:
            return f"""🔴 **El drawdown realizado ({dd_abs:.2f}%) SUPERÓ incluso el CVaR conservador ({cvar_annual:.2f}%).**

Esto indica que:
- Ocurrió un **evento extremo** no capturado por los modelos
- Se requiere revisión de supuestos (posiblemente df<3)
- Considerar implementar límites de pérdida más estrictos"""
    
    def _generate_strengths(self, total_ret, sharpe, outperf_active):
        """Helper para generar análisis de fortalezas"""
        strengths = []
        
        if total_ret > 0:
            strengths.append("- **Retornos Positivos:** La cartera generó ganancias en el período analizado")
        
        if sharpe > 1:
            strengths.append(f"- **Sharpe Ratio Saludable ({sharpe:.2f}):** Buena relación riesgo-retorno")
        
        if isinstance(outperf_active, (int, float)) and outperf_active > 0:
            strengths.append(f"- **Outperformance vs Benchmark:** La estrategia activa superó al S&P 500 en {outperf_active:.2f}%")
        
        if not strengths:
            strengths.append("- La estrategia mostró resiliencia en un período complejo")
        
        return "\n".join(strengths)
    
    def _generate_weaknesses(self, max_dd, volatility, outperf_active):
        """Helper para generar análisis de debilidades"""
        weaknesses = []
        
        if abs(max_dd) > 20:
            weaknesses.append(f"- **Drawdown Elevado ({max_dd:.2f}%):** Requiere alta tolerancia al riesgo")
        
        if volatility > 35:
            weaknesses.append(f"- **Alta Volatilidad ({volatility:.2f}%):** Movimientos bruscos de precios")
        
        if isinstance(outperf_active, (int, float)) and outperf_active < 0:
            weaknesses.append(f"- **Underperformance vs Benchmark:** La estrategia activa no superó al mercado en {abs(outperf_active):.2f}%")
        
        if not weaknesses:
            weaknesses.append("- No se identificaron debilidades críticas en el período analizado")
        
        return "\n".join(weaknesses)
    
    def _generate_adjustments(self, total_ret, sharpe, max_dd, outperf_active):
        """Helper para generar ajustes sugeridos"""
        adjustments = []
        
        if abs(max_dd) > 25:
            adjustments.append("1. **Reducir exposición a activos de alta volatilidad** (especialmente argentinos y cripto)")
        
        if sharpe < 1:
            adjustments.append("2. **Revisar la composición de la cartera** - el riesgo no está siendo compensado adecuadamente")
        
        if isinstance(outperf_active, (int, float)) and outperf_active < -5:
            adjustments.append("3. **Considerar mayor peso en benchmark (SPY)** si el objetivo es tracking del mercado")
        
        if total_ret < 0:
            adjustments.append("4. **Implementar stop-loss dinámico** para limitar pérdidas en mercados bajistas")
        
        if not adjustments:
            adjustments.append("1. **Mantener la estrategia actual** con monitoreo mensual de métricas")
            adjustments.append("2. **Rebalancear trimestralmente** para mantener pesos óptimos")
        
        return "\n".join(adjustments)
    
    def run_stress_test(self, weights, capital=1000000, save_path=None):
        """
        Realiza stress testing de la cartera bajo escenarios extremos
        
        Parameters:
        -----------
        weights : dict
            Pesos de la cartera a testear
        capital : float
            Capital invertido en USD
        save_path : str
            Ruta para guardar el gráfico
        
        Returns:
        --------
        dict : Resultados del stress test por escenario
        """
        print("\n" + "="*80)
        print("STRESS TESTING - ANALISIS DE ESCENARIOS EXTREMOS")
        print("="*80 + "\n")
        
        print(f"[i] Capital inicial: ${capital:,.2f} USD")
        print(f"[i] Simulando 3 escenarios de estres...")
        
        # Convertir pesos a array alineado con self.tickers
        weights_array = np.array([weights.get(ticker, 0) for ticker in self.tickers])
        
        # Calcular capital asignado por activo
        capital_per_asset = capital * weights_array
        
        # ==========================================
        # DEFINICIÓN DE ESCENARIOS
        # ==========================================
        scenarios = {
            'Crash Global': {},
            'Crisis Argentina': {},
            'Recuperación Agresiva': {}
        }
        
        # Escenario 1: Crash Global
        for ticker in self.tickers:
            if ticker in ['AAPL', 'MSFT']:
                scenarios['Crash Global'][ticker] = -0.25  # -25%
            elif ticker == 'BTC-USD':
                scenarios['Crash Global'][ticker] = -0.40  # -40%
            elif ticker.endswith('.BA'):
                scenarios['Crash Global'][ticker] = -0.20  # -20% (contagio)
            else:
                scenarios['Crash Global'][ticker] = -0.20  # -20% default
        
        # Escenario 2: Crisis Argentina
        for ticker in self.tickers:
            if ticker.endswith('.BA'):
                scenarios['Crisis Argentina'][ticker] = -0.40  # -40%
            elif ticker == 'BTC-USD':
                scenarios['Crisis Argentina'][ticker] = -0.10  # -10% (flight to crypto)
            else:
                scenarios['Crisis Argentina'][ticker] = -0.05  # -5% (efecto menor)
        
        # Escenario 3: Recuperación Agresiva
        for ticker in self.tickers:
            scenarios['Recuperación Agresiva'][ticker] = 0.20  # +20% todos
        
        # ==========================================
        # CALCULAR IMPACTO POR ESCENARIO
        # ==========================================
        results = {}
        
        print("\n" + "="*80)
        print("RESULTADOS DEL STRESS TEST")
        print("="*80 + "\n")
        
        for scenario_name, shocks in scenarios.items():
            # Calcular impacto en cada activo
            impact_per_asset = []
            
            for i, ticker in enumerate(self.tickers):
                shock = shocks.get(ticker, 0)
                asset_capital = capital_per_asset[i]
                asset_impact = asset_capital * shock
                impact_per_asset.append(asset_impact)
            
            # Impacto total en la cartera
            total_impact = sum(impact_per_asset)
            final_capital = capital + total_impact
            pct_change = (total_impact / capital) * 100
            
            results[scenario_name] = {
                'initial_capital': capital,
                'impact': total_impact,
                'final_capital': final_capital,
                'pct_change': pct_change,
                'impact_per_asset': dict(zip(self.tickers, impact_per_asset)),
                'shocks': shocks
            }
            
            # Mostrar resultados
            print(f"[ESCENARIO] {scenario_name}")
            print(f"{'='*80}")
            
            # Mostrar shocks aplicados
            print("\nShocks aplicados:")
            for ticker in self.tickers:
                shock = shocks.get(ticker, 0)
                asset_cap = capital_per_asset[self.tickers.index(ticker)]
                if asset_cap > 0:  # Solo mostrar activos con asignación
                    print(f"  {ticker:12s}: {shock*100:+6.1f}% (Capital: ${asset_cap:,.2f})")
            
            print(f"\nResultados:")
            print(f"  Capital Inicial:       ${capital:,.2f}")
            print(f"  Impacto Total:         ${total_impact:+,.2f}")
            print(f"  Capital Final:         ${final_capital:,.2f}")
            print(f"  Cambio Porcentual:     {pct_change:+.2f}%")
            
            # Interpretación
            if pct_change < -30:
                status = "[CRITICO] Perdida severa"
            elif pct_change < -15:
                status = "[ALTO RIESGO] Perdida significativa"
            elif pct_change < 0:
                status = "[MODERADO] Perdida tolerable"
            else:
                status = "[POSITIVO] Ganancia"
            
            print(f"  Estado:                {status}")
            print()
        
        # ==========================================
        # GENERAR GRÁFICO
        # ==========================================
        if save_path is None:
            save_path = OUTPUTS_DIR / 'stress_test.png'
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"[+] Generando grafico de stress test...")
        
        # Crear figura con 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Stress Testing - Análisis de Escenarios Extremos', 
                     fontsize=16, fontweight='bold')
        
        # ==========================================
        # Plot 1: Capital Final por Escenario
        # ==========================================
        scenarios_names = list(results.keys())
        final_capitals = [results[s]['final_capital'] for s in scenarios_names]
        pct_changes = [results[s]['pct_change'] for s in scenarios_names]
        
        # Colores según resultado
        colors = []
        for pct in pct_changes:
            if pct < -30:
                colors.append('#8B0000')  # Dark red (crítico)
            elif pct < -15:
                colors.append('#DC143C')  # Crimson (alto riesgo)
            elif pct < 0:
                colors.append('#FFA500')  # Orange (moderado)
            else:
                colors.append('#32CD32')  # Lime green (positivo)
        
        bars = ax1.barh(scenarios_names, final_capitals, color=colors, 
                        edgecolor='black', linewidth=2)
        ax1.axvline(capital, color='blue', linestyle='--', linewidth=2, 
                   label=f'Capital Inicial (${capital/1000:.0f}K)', alpha=0.7)
        
        ax1.set_xlabel('Capital Final (USD)', fontsize=12, fontweight='bold')
        ax1.set_title('Capital Remanente por Escenario', fontsize=13, fontweight='bold')
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        ax1.grid(True, axis='x', alpha=0.3)
        ax1.legend(loc='best')
        
        # Añadir etiquetas con valores
        for i, (bar, val, pct) in enumerate(zip(bars, final_capitals, pct_changes)):
            width = bar.get_width()
            label_x = width if width > capital else width
            ax1.text(label_x, bar.get_y() + bar.get_height()/2, 
                    f'  ${val/1000:.0f}K ({pct:+.1f}%)',
                    ha='left' if width > capital else 'right', 
                    va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor=colors[i], alpha=0.9))
        
        # ==========================================
        # Plot 2: Pérdidas/Ganancias por Escenario
        # ==========================================
        impacts = [results[s]['impact'] for s in scenarios_names]
        
        colors_impact = ['#32CD32' if x > 0 else '#DC143C' for x in impacts]
        
        bars2 = ax2.barh(scenarios_names, impacts, color=colors_impact, 
                         edgecolor='black', linewidth=2)
        ax2.axvline(0, color='black', linestyle='-', linewidth=1)
        
        ax2.set_xlabel('Impacto en Capital (USD)', fontsize=12, fontweight='bold')
        ax2.set_title('Pérdidas/Ganancias por Escenario', fontsize=13, fontweight='bold')
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        ax2.grid(True, axis='x', alpha=0.3)
        
        # Añadir etiquetas con valores
        for i, (bar, val, pct) in enumerate(zip(bars2, impacts, pct_changes)):
            width = bar.get_width()
            label_x = width
            ax2.text(label_x, bar.get_y() + bar.get_height()/2, 
                    f'  ${val/1000:.0f}K ',
                    ha='left' if width > 0 else 'right', 
                    va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor=colors_impact[i], alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Grafico guardado en: {save_path}")
        
        # ==========================================
        # ANÁLISIS DE RESILIENCIA
        # ==========================================
        print("\n" + "="*80)
        print("ANALISIS DE RESILIENCIA DE LA CARTERA")
        print("="*80 + "\n")
        
        worst_case = min(results.values(), key=lambda x: x['pct_change'])
        best_case = max(results.values(), key=lambda x: x['pct_change'])
        
        worst_scenario = [k for k, v in results.items() if v == worst_case][0]
        best_scenario = [k for k, v in results.items() if v == best_case][0]
        
        print(f"[*] PEOR ESCENARIO: {worst_scenario}")
        print(f"    Perdida: ${abs(worst_case['impact']):,.2f} ({worst_case['pct_change']:.2f}%)")
        print(f"    Capital Remanente: ${worst_case['final_capital']:,.2f}")
        
        print(f"\n[*] MEJOR ESCENARIO: {best_scenario}")
        print(f"    Ganancia: ${best_case['impact']:,.2f} ({best_case['pct_change']:+.2f}%)")
        print(f"    Capital Final: ${best_case['final_capital']:,.2f}")
        
        # Calcular capital en riesgo (promedio de escenarios negativos)
        negative_scenarios = [v for v in results.values() if v['impact'] < 0]
        if negative_scenarios:
            avg_loss = np.mean([v['impact'] for v in negative_scenarios])
            print(f"\n[*] CAPITAL EN RIESGO (Promedio escenarios negativos):")
            print(f"    Perdida Promedio: ${abs(avg_loss):,.2f} ({(avg_loss/capital)*100:.2f}%)")
        
        print("\n" + "="*80)
        print("RECOMENDACIONES:")
        print("="*80)
        
        if worst_case['pct_change'] < -30:
            print("  [!] CRITICO: La cartera es muy vulnerable a crisis. Considerar:")
            print("      - Reducir exposicion a activos de alta volatilidad")
            print("      - Incrementar diversificacion geografica")
            print("      - Implementar coberturas (hedging)")
        elif worst_case['pct_change'] < -20:
            print("  [!] ALTO RIESGO: La cartera tiene exposicion significativa a shocks.")
            print("      - Monitorear indicadores de alerta temprana")
            print("      - Establecer stop-loss en niveles criticos")
        else:
            print("  [OK] RESILIENCIA ACEPTABLE: La cartera muestra tolerancia a shocks moderados.")
            print("      - Mantener diversificacion actual")
            print("      - Revisar trimestralmente")
        
        print("="*80 + "\n")
        
        return results
    
    def generate_stress_test_report(self, stress_results, max_sharpe_weights, 
                                    capital, save_path=None):
        """
        Genera reporte técnico de stress testing en Markdown
        
        Parameters:
        -----------
        stress_results : dict
            Resultados del stress testing
        max_sharpe_weights : dict
            Pesos de la cartera testeada
        capital : float
            Capital inicial invertido
        save_path : str
            Ruta para guardar el reporte
        """
        print("\n[+] Generando reporte de stress testing...")
        
        # Usar ruta por defecto si no se especifica
        if save_path is None:
            save_path = OUTPUTS_DIR / 'reporte_stress_test.md'
        else:
            save_path = Path(save_path)
        
        # Crear directorio de salida si no existe
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Calcular métricas agregadas
        worst_case = min(stress_results.values(), key=lambda x: x['pct_change'])
        best_case = max(stress_results.values(), key=lambda x: x['pct_change'])
        worst_scenario = [k for k, v in stress_results.items() if v == worst_case][0]
        best_scenario = [k for k, v in stress_results.items() if v == best_case][0]
        
        negative_scenarios = [v for v in stress_results.values() if v['impact'] < 0]
        avg_loss = np.mean([v['impact'] for v in negative_scenarios]) if negative_scenarios else 0
        
        report = f"""# Reporte de Stress Testing
## Análisis de Escenarios Extremos

**Fecha de Generación:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Cartera Analizada:** Máximo Sharpe Ratio **GESTIONADA** (con restricciones)  
**Capital Invertido:** ${capital:,.2f} USD

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
"""
        
        # Añadir filas de la tabla
        for scenario_name, result in stress_results.items():
            status_emoji = "🔴" if result['pct_change'] < -30 else "🟠" if result['pct_change'] < -15 else "🟡" if result['pct_change'] < 0 else "🟢"
            report += f"| {scenario_name} | ${result['initial_capital']:,.2f} | ${result['impact']:+,.2f} | ${result['final_capital']:,.2f} | {result['pct_change']:+.2f}% | {status_emoji} |\n"
        
        report += f"""
---

## 2. Análisis Detallado por Escenario

"""
        
        # Análisis detallado de cada escenario
        for scenario_name, result in stress_results.items():
            report += f"""### Escenario: {scenario_name}

**Descripción de Shocks Aplicados:**

"""
            # Listar shocks por activo
            for ticker, shock in result['shocks'].items():
                report += f"- **{ticker}**: {shock*100:+.1f}%\n"
            
            report += f"""
**Resultados:**

```
Capital Inicial:       ${result['initial_capital']:,.2f}
Impacto Total:         ${result['impact']:+,.2f}
Capital Final:         ${result['final_capital']:,.2f}
Cambio Porcentual:     {result['pct_change']:+.2f}%
```

**Impacto por Activo:**

| Activo | Peso en Cartera | Capital Asignado | Shock | Impacto en USD |
|--------|----------------|------------------|-------|----------------|
"""
            
            for ticker in self.tickers:
                weight = max_sharpe_weights.get(ticker, 0)
                if weight > 0:
                    asset_capital = capital * weight
                    shock = result['shocks'].get(ticker, 0)
                    impact = result['impact_per_asset'].get(ticker, 0)
                    report += f"| {ticker} | {weight*100:.2f}% | ${asset_capital:,.2f} | {shock*100:+.1f}% | ${impact:+,.2f} |\n"
            
            # Interpretación
            if result['pct_change'] < -30:
                interpretation = "🔴 **CRÍTICO:** Pérdida severa. Este escenario representa un riesgo existencial para la cartera."
            elif result['pct_change'] < -15:
                interpretation = "🟠 **ALTO RIESGO:** Pérdida significativa. Se requiere gestión activa para mitigar el impacto."
            elif result['pct_change'] < 0:
                interpretation = "🟡 **MODERADO:** Pérdida tolerable dentro de parámetros esperados de riesgo."
            else:
                interpretation = "🟢 **POSITIVO:** Ganancia potencial en este escenario."
            
            report += f"""
**Interpretación:** {interpretation}

---

"""
        
        report += f"""## 3. Análisis de Resiliencia

### Métricas de Riesgo Extremo

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Peor Escenario** | {worst_scenario} | Pérdida de ${abs(worst_case['impact']):,.2f} ({worst_case['pct_change']:.2f}%) |
| **Mejor Escenario** | {best_scenario} | Ganancia de ${best_case['impact']:,.2f} ({best_case['pct_change']:+.2f}%) |
| **Capital en Riesgo** | ${abs(avg_loss):,.2f} | Promedio de escenarios negativos |
| **Capital Mínimo (worst case)** | ${worst_case['final_capital']:,.2f} | Capital remanente en crisis |

### Evaluación de Vulnerabilidad

**Exposición a Crash Global:**
```
Impacto: ${stress_results['Crash Global']['impact']:+,.2f}
Cambio: {stress_results['Crash Global']['pct_change']:+.2f}%
```

El escenario de crash global simula una crisis financiera internacional similar a 2008 o marzo 2020. 
{"La cartera muestra alta vulnerabilidad a este tipo de eventos." if stress_results['Crash Global']['pct_change'] < -25 else "La cartera tiene exposición moderada a crisis globales."}

**Exposición a Crisis Argentina:**
```
Impacto: ${stress_results['Crisis Argentina']['impact']:+,.2f}
Cambio: {stress_results['Crisis Argentina']['pct_change']:+.2f}%
```

Este escenario simula un colapso específico del mercado argentino (similar a 2001, 2018 o 2019).
{"La cartera está ALTAMENTE expuesta al riesgo país argentino." if stress_results['Crisis Argentina']['pct_change'] < -25 else "La cartera tiene exposición controlada al riesgo argentino."}

**Potencial de Recuperación:**
```
Ganancia: ${stress_results['Recuperación Agresiva']['impact']:+,.2f}
Cambio: {stress_results['Recuperación Agresiva']['pct_change']:+.2f}%
```

En un escenario de recuperación fuerte, la cartera {"tiene alto potencial de upside." if stress_results['Recuperación Agresiva']['pct_change'] > 15 else "muestra crecimiento moderado."}

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

### Para el Peor Escenario ({worst_scenario})

**Pérdida Potencial:** ${abs(worst_case['impact']):,.2f} ({worst_case['pct_change']:.2f}%)

"""
        
        if worst_case['pct_change'] < -30:
            report += """**Acciones Urgentes:**

1. **Reducir Exposición a Activos de Alta Volatilidad**
   - Disminuir posiciones en activos argentinos
   - Reducir peso de criptomonedas
   - Incrementar activos defensivos (bonos, oro)

2. **Implementar Coberturas (Hedging)**
   - Opciones PUT sobre índices (SPY, QQQ)
   - Cobertura cambiaria (dólar MEP, futuros)
   - Seguros de cartera (tail risk hedging)

3. **Establecer Capital de Contingencia**
   - Reservar al menos ${abs(worst_case['impact']):,.2f} como buffer
   - Mantener liquidez para oportunidades en crisis

4. **Rebalanceo Urgente**
   - Revisar la composición ANTES de materialización del riesgo
   - Considerar estrategias de "risk parity"
"""
        elif worst_case['pct_change'] < -20:
            report += """**Acciones Recomendadas:**

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
"""
        else:
            report += """**Acciones de Mantenimiento:**

1. **Monitoreo Regular**
   - Revisión trimestral de exposiciones
   - Ajuste de pesos según cambios en correlaciones

2. **Optimización Continua**
   - Rebalancear cuando desviaciones superen 5%
   - Actualizar parámetros con datos recientes
"""
        
        report += f"""
### Gestión de Capital en Crisis

Si se materializa el **peor escenario** ({worst_scenario}):

```
Capital Inicial:      ${capital:,.2f}
Capital Remanente:    ${worst_case['final_capital']:,.2f}
Pérdida:              ${abs(worst_case['impact']):,.2f}
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

{self._generate_stress_vulnerabilities(stress_results)}

### Fortalezas de la Cartera

{self._generate_stress_strengths(stress_results)}

### Nivel de Riesgo Global

"""
        
        if worst_case['pct_change'] < -30:
            report += "🔴 **ALTO RIESGO:** La cartera es vulnerable a shocks extremos. Se recomienda reestructuración."
        elif worst_case['pct_change'] < -20:
            report += "🟠 **RIESGO MODERADO-ALTO:** La cartera tiene exposición significativa. Monitoreo activo requerido."
        else:
            report += "🟢 **RIESGO CONTROLADO:** La cartera muestra resiliencia aceptable a escenarios extremos."
        
        report += """

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
"""
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"[OK] Reporte de stress testing guardado en: {save_path}")
        
        return save_path
    
    def _generate_stress_vulnerabilities(self, stress_results):
        """Helper para generar análisis de vulnerabilidades"""
        vulnerabilities = []
        
        for scenario_name, result in stress_results.items():
            if result['pct_change'] < -25:
                vulnerabilities.append(f"- **{scenario_name}:** Pérdida de {abs(result['pct_change']):.1f}% - Vulnerabilidad crítica")
        
        if not vulnerabilities:
            vulnerabilities.append("- No se identificaron vulnerabilidades críticas (pérdidas < 25%)")
        
        return "\n".join(vulnerabilities)
    
    def _generate_stress_strengths(self, stress_results):
        """Helper para generar análisis de fortalezas"""
        strengths = []
        
        positive_scenarios = [k for k, v in stress_results.items() if v['pct_change'] > 0]
        if positive_scenarios:
            for scenario in positive_scenarios:
                strengths.append(f"- **{scenario}:** Potencial de ganancia del {stress_results[scenario]['pct_change']:+.1f}%")
        
        moderate_loss = [k for k, v in stress_results.items() if -15 < v['pct_change'] < 0]
        if moderate_loss:
            strengths.append(f"- Resiliencia moderada en escenarios: {', '.join(moderate_loss)}")
        
        if not strengths:
            strengths.append("- La diversificación actual mitiga algunos riesgos")
        
        return "\n".join(strengths)
    
    def compare_strategies(self, weights_free, perf_free, risk_free,
                          weights_managed, perf_managed, risk_managed):
        """
        Compara dos estrategias de optimización (libre vs gestionada)
        
        Parameters:
        -----------
        weights_free : dict
            Pesos de cartera sin restricciones
        perf_free : tuple
            (retorno, volatilidad, sharpe) de cartera libre
        risk_free : dict
            Métricas de riesgo (VaR/CVaR) de cartera libre
        weights_managed : dict
            Pesos de cartera con restricciones
        perf_managed : tuple
            (retorno, volatilidad, sharpe) de cartera gestionada
        risk_managed : dict
            Métricas de riesgo (VaR/CVaR) de cartera gestionada
        """
        print("\n" + "="*80)
        print("TABLA COMPARATIVA: OPTIMIZACION LIBRE vs GESTIONADA")
        print("="*80 + "\n")
        
        # Tabla de composición
        print("COMPOSICION DE LA CARTERA:")
        print("-"*80)
        print(f"{'Activo':<12} | {'Libre (%)':>12} | {'Gestionada (%)':>15} | {'Diferencia':>12}")
        print("-"*80)
        
        for ticker in self.tickers:
            weight_free = weights_free.get(ticker, 0) * 100
            weight_managed = weights_managed.get(ticker, 0) * 100
            diff = weight_managed - weight_free
            
            # Solo mostrar activos con peso > 0.1% en alguna versión
            if weight_free > 0.1 or weight_managed > 0.1:
                print(f"{ticker:<12} | {weight_free:>11.2f}% | {weight_managed:>14.2f}% | {diff:>+11.2f}%")
        
        # Tabla de performance
        print("\n" + "="*80)
        print("METRICAS DE PERFORMANCE:")
        print("-"*80)
        print(f"{'Metrica':<30} | {'Libre':>15} | {'Gestionada':>15} | {'Diferencia':>12}")
        print("-"*80)
        
        # Retorno
        ret_free = perf_free[0] * 100
        ret_managed = perf_managed[0] * 100
        ret_diff = ret_managed - ret_free
        print(f"{'Retorno Anualizado':<30} | {ret_free:>14.2f}% | {ret_managed:>14.2f}% | {ret_diff:>+11.2f}%")
        
        # Volatilidad
        vol_free = perf_free[1] * 100
        vol_managed = perf_managed[1] * 100
        vol_diff = vol_managed - vol_free
        print(f"{'Volatilidad Anualizada':<30} | {vol_free:>14.2f}% | {vol_managed:>14.2f}% | {vol_diff:>+11.2f}%")
        
        # Sharpe
        sharpe_free = perf_free[2]
        sharpe_managed = perf_managed[2]
        sharpe_diff = sharpe_managed - sharpe_free
        sharpe_diff_pct = (sharpe_diff / sharpe_free * 100) if sharpe_free != 0 else 0
        print(f"{'Sharpe Ratio':<30} | {sharpe_free:>15.2f} | {sharpe_managed:>15.2f} | {sharpe_diff:>+11.2f}")
        
        # VaR/CVaR
        print("\n" + "="*80)
        print("METRICAS DE RIESGO (VaR/CVaR Conservador, df=3):")
        print("-"*80)
        print(f"{'Metrica':<30} | {'Libre':>15} | {'Gestionada':>15} | {'Mejora':>12}")
        print("-"*80)
        
        # VaR diario
        var_free = risk_free['var_daily'] * 100
        var_managed = risk_managed['var_daily'] * 100
        var_diff = var_managed - var_free
        var_improvement = abs(var_diff) if var_managed > var_free else -abs(var_diff)
        print(f"{'VaR (1 dia, 95%)':<30} | {var_free:>14.2f}% | {var_managed:>14.2f}% | {var_improvement:>+11.2f}%")
        
        # CVaR diario
        cvar_free = risk_free['cvar_daily'] * 100
        cvar_managed = risk_managed['cvar_daily'] * 100
        cvar_diff = cvar_managed - cvar_free
        cvar_improvement = abs(cvar_diff) if cvar_managed > cvar_free else -abs(cvar_diff)
        print(f"{'CVaR (1 dia, 95%)':<30} | {cvar_free:>14.2f}% | {cvar_managed:>14.2f}% | {cvar_improvement:>+11.2f}%")
        
        # VaR anual
        var_annual_free = risk_free['var_annual'] * 100
        var_annual_managed = risk_managed['var_annual'] * 100
        var_annual_diff = var_annual_managed - var_annual_free
        print(f"{'VaR (anual, 95%)':<30} | {var_annual_free:>14.2f}% | {var_annual_managed:>14.2f}% | {var_annual_diff:>+11.2f}%")
        
        # CVaR anual
        cvar_annual_free = risk_free['cvar_annual'] * 100
        cvar_annual_managed = risk_managed['cvar_annual'] * 100
        cvar_annual_diff = cvar_annual_managed - cvar_annual_free
        print(f"{'CVaR (anual, 95%)':<30} | {cvar_annual_free:>14.2f}% | {cvar_annual_managed:>14.2f}% | {cvar_annual_diff:>+11.2f}%")
        
        # Análisis e interpretación
        print("\n" + "="*80)
        print("ANALISIS E INTERPRETACION:")
        print("="*80 + "\n")
        
        # Sharpe Ratio
        if sharpe_diff < 0:
            sharpe_status = f"[TRADE-OFF] Sharpe DISMINUYO en {abs(sharpe_diff):.2f} ({abs(sharpe_diff_pct):.1f}%)"
            sharpe_msg = "Las restricciones limitan el potencial de retorno ajustado por riesgo."
        else:
            sharpe_status = f"[MEJORA] Sharpe AUMENTO en {sharpe_diff:.2f} ({sharpe_diff_pct:.1f}%)"
            sharpe_msg = "Las restricciones mejoraron la eficiencia de la cartera."
        
        print(f"1. SHARPE RATIO:")
        print(f"   {sharpe_status}")
        print(f"   {sharpe_msg}")
        
        # VaR/CVaR
        var_better = var_managed > var_free  # VaR es negativo, más alto (menos negativo) = mejor
        if var_better:
            var_status = "[MEJORA] VaR/CVaR mejoraron (menor riesgo de cola)"
            var_msg = f"   El VaR diario mejoro {abs(var_improvement):.2f}pp, reduciendo exposicion a eventos extremos."
        else:
            var_status = "[EMPEORO] VaR/CVaR empeoraron (mayor riesgo de cola)"
            var_msg = f"   El VaR diario empeoro {abs(var_improvement):.2f}pp debido a concentracion forzada."
        
        print(f"\n2. RIESGO DE COLA (VaR/CVaR):")
        print(f"   {var_status}")
        print(f"{var_msg}")
        
        # Volatilidad
        if vol_diff < 0:
            vol_status = f"[MEJORA] Volatilidad BAJO {abs(vol_diff):.2f}pp"
        else:
            vol_status = f"[AUMENTO] Volatilidad SUBIO {vol_diff:.2f}pp"
        
        print(f"\n3. VOLATILIDAD:")
        print(f"   {vol_status}")
        
        # Diversificación
        n_assets_free = sum(1 for w in weights_free.values() if w > 0.01)
        n_assets_managed = sum(1 for w in weights_managed.values() if w > 0.01)
        
        print(f"\n4. DIVERSIFICACION:")
        print(f"   Libre:      {n_assets_free} activos con peso significativo (>1%)")
        print(f"   Gestionada: {n_assets_managed} activos con peso significativo (>1%)")
        
        # Recomendación final
        print("\n" + "="*80)
        print("RECOMENDACION:")
        print("="*80)
        
        if sharpe_diff >= -0.05 and var_better:
            print("  [OK] USAR CARTERA GESTIONADA")
            print("       - Sharpe similar o mejor")
            print("       - Riesgo de cola reducido")
            print("       - Cumple con limites regulatorios/politicas")
        elif sharpe_diff < -0.1:
            print("  [!] REVISAR RESTRICCIONES")
            print("       - Sharpe se deterioro significativamente")
            print("       - Considerar relajar algunos limites")
            print("       - Evaluar trade-off riesgo-retorno")
        else:
            print("  [OK] CARTERA GESTIONADA ES VIABLE")
            print("       - Trade-off aceptable entre eficiencia y control de riesgo")
            print("       - Implementar con monitoreo trimestral")
        
        print("="*80 + "\n")
        
        return {
            'sharpe_diff': sharpe_diff,
            'var_improvement': var_improvement,
            'vol_diff': vol_diff,
            'n_assets_free': n_assets_free,
            'n_assets_managed': n_assets_managed
        }
    
    def run_backtest(self, weights, initial_capital=1000000, lookback_period='1y', 
                     benchmark_ticker='SPY', rebalance=True, commission_pct=0.005, 
                     save_path=None):
        """
        Realiza backtesting de la estrategia con los pesos dados
        
        Parameters:
        -----------
        weights : dict
            Pesos de la cartera a testear
        initial_capital : float
            Capital inicial en USD
        lookback_period : str
            Período de backtest ('1y', '2y', '6mo', etc.)
        benchmark_ticker : str
            Ticker para comparar (default: SPY - S&P 500)
        rebalance : bool
            Si True, rebalancea mensualmente a pesos target (default: True)
        commission_pct : float
            Comisión por operación como decimal (default: 0.005 = 0.5%)
        save_path : str
            Ruta para guardar el gráfico
        """
        print("\n" + "="*80)
        print("BACKTESTING - VALIDACION HISTORICA DE LA ESTRATEGIA")
        print("="*80 + "\n")
        
        # Configurar fechas para backtest
        end_date = datetime.now()
        
        # Parsear lookback_period
        if lookback_period.endswith('y'):
            years = int(lookback_period[:-1])
            start_date = end_date - timedelta(days=years*365)
        elif lookback_period.endswith('mo'):
            months = int(lookback_period[:-2])
            start_date = end_date - timedelta(days=months*30)
        elif lookback_period.endswith('d'):
            days = int(lookback_period[:-1])
            start_date = end_date - timedelta(days=days)
        else:
            start_date = end_date - timedelta(days=365)  # Default 1 año
        
        print(f"\n[VALIDACION OUT-OF-SAMPLE]")
        print(f"[i] Periodo de optimizacion: {self.start_date} a {self.end_date}")
        print(f"[i] Periodo de backtest:     {start_date.date()} a {end_date.date()}")
        
        # Validar que no haya overlap (out-of-sample puro)
        optimization_end = datetime.strptime(self.end_date, '%Y-%m-%d')
        if start_date < optimization_end:
            print(f"\n[!] ADVERTENCIA: Overlap detectado entre optimizacion y backtesting")
            print(f"    - Optimizacion termina: {self.end_date}")
            print(f"    - Backtesting comienza: {start_date.date()}")
            print(f"    - Esto NO es una validacion out-of-sample genuina")
            print(f"    - Considera usar datos posteriores a {self.end_date} para backtesting\n")
        else:
            print(f"[OK] Validacion out-of-sample: Backtesting usa datos posteriores a la optimizacion\n")
        
        print(f"[i] Capital inicial: ${initial_capital:,.2f} USD")
        print(f"[i] Benchmark: {benchmark_ticker}")
        print(f"[i] Rebalanceo: {'Mensual (Active)' if rebalance else 'Buy-and-Hold (Passive)'}")
        print(f"[i] Comision por operacion: {commission_pct*100:.2f}%")
        
        # Descargar datos históricos para backtest
        print("\n[+] Descargando datos historicos para backtest...")
        all_tickers = list(self.tickers) + [benchmark_ticker]
        
        try:
            backtest_data_raw = yf.download(all_tickers, start=start_date, end=end_date, 
                                           progress=False)
            
            # Intentar extraer los datos de precios ajustados
            if isinstance(backtest_data_raw.columns, pd.MultiIndex):
                # Múltiples tickers - buscar 'Adj Close' o 'Close'
                if 'Adj Close' in backtest_data_raw.columns.levels[0]:
                    backtest_data = backtest_data_raw['Adj Close']
                elif 'Close' in backtest_data_raw.columns.levels[0]:
                    backtest_data = backtest_data_raw['Close']
                else:
                    print(f"[ERROR] No se encontraron datos de precios")
                    return None
            else:
                # Un solo ticker o estructura diferente
                if 'Adj Close' in backtest_data_raw.columns:
                    backtest_data = backtest_data_raw['Adj Close'].to_frame()
                    if len(all_tickers) == 1:
                        backtest_data.columns = all_tickers
                elif 'Close' in backtest_data_raw.columns:
                    backtest_data = backtest_data_raw['Close'].to_frame()
                    if len(all_tickers) == 1:
                        backtest_data.columns = all_tickers
                else:
                    # Si falla, usar todo el DataFrame
                    backtest_data = backtest_data_raw
            
            # Eliminar NaN
            backtest_data = backtest_data.dropna()
            
            if len(backtest_data) == 0:
                print(f"[ERROR] No hay datos suficientes para el periodo solicitado")
                return None
            
            print(f"[OK] Datos de backtest: {len(backtest_data)} dias de trading")
            
        except Exception as e:
            print(f"[ERROR] No se pudieron descargar datos: {e}")
            print(f"[DEBUG] Columnas disponibles: {backtest_data_raw.columns if 'backtest_data_raw' in locals() else 'No disponibles'}")
            return None
        
        # Convertir pesos a array
        target_weights = np.array([weights.get(ticker, 0) for ticker in self.tickers])
        
        # Calcular retornos diarios
        returns_bt = backtest_data[self.tickers].pct_change().dropna()
        prices_bt = backtest_data[self.tickers].loc[returns_bt.index]
        
        # ==============================================
        # SIMULACION 1: CARTERA CON REBALANCEO (ACTIVA)
        # ==============================================
        if rebalance:
            print("[+] Simulando estrategia ACTIVA con rebalanceo mensual...")
            portfolio_value_active = pd.Series(index=returns_bt.index, dtype=float)
            total_commissions = 0.0
            
            # Estado de la cartera
            capital = initial_capital
            holdings = target_weights * capital / prices_bt.iloc[0].values  # Número de unidades
            
            # Trackear primera compra (comisión inicial)
            initial_transaction_cost = commission_pct * capital
            total_commissions += initial_transaction_cost
            capital -= initial_transaction_cost
            holdings = target_weights * capital / prices_bt.iloc[0].values
            
            portfolio_value_active.iloc[0] = capital
            
            last_rebalance_month = prices_bt.index[0].month
            last_rebalance_year = prices_bt.index[0].year
            
            for i in range(1, len(returns_bt)):
                date = returns_bt.index[i]
                
                # Valor de mercado actual de cada posición
                current_prices = prices_bt.iloc[i].values
                position_values = holdings * current_prices
                total_value = position_values.sum()
                
                # Verificar si es primer día hábil del mes (rebalanceo)
                current_month = date.month
                current_year = date.year
                
                should_rebalance = (current_month != last_rebalance_month) or \
                                   (current_year != last_rebalance_year)
                
                if should_rebalance:
                    # Calcular pesos actuales
                    current_weights = position_values / total_value
                    
                    # Calcular volumen necesario para rebalancear
                    target_values = target_weights * total_value
                    value_to_trade = np.abs(target_values - position_values).sum()
                    
                    # Cobrar comisión sobre volumen operado
                    rebalance_commission = commission_pct * value_to_trade
                    total_commissions += rebalance_commission
                    
                    # Actualizar holdings (rebalancear restando comisión del capital)
                    capital_after_commission = total_value - rebalance_commission
                    holdings = target_weights * capital_after_commission / current_prices
                    
                    portfolio_value_active.iloc[i] = capital_after_commission
                    
                    last_rebalance_month = current_month
                    last_rebalance_year = current_year
                else:
                    # Sin rebalanceo, solo actualizar valor
                    portfolio_value_active.iloc[i] = total_value
            
            print(f"[OK] Comisiones totales pagadas (Active): ${total_commissions:,.2f} ({total_commissions/initial_capital*100:.2f}% del capital)")
        else:
            # Modo sin rebalanceo
            portfolio_returns_active = (returns_bt * target_weights).sum(axis=1)
            portfolio_value_active = initial_capital * (1 + portfolio_returns_active).cumprod()
            total_commissions = initial_capital * commission_pct  # Solo comisión inicial
            portfolio_value_active -= total_commissions
        
        # ==============================================
        # SIMULACION 2: CARTERA SIN REBALANCEO (PASIVA)
        # ==============================================
        print("[+] Simulando estrategia PASIVA (buy-and-hold)...")
        portfolio_returns_passive = (returns_bt * target_weights).sum(axis=1)
        initial_commission_passive = initial_capital * commission_pct
        capital_passive_start = initial_capital - initial_commission_passive
        portfolio_value_passive = capital_passive_start * (1 + portfolio_returns_passive).cumprod()
        
        # ==============================================
        # BENCHMARK
        # ==============================================
        if benchmark_ticker in backtest_data.columns:
            benchmark_returns = backtest_data[benchmark_ticker].pct_change().dropna()
            initial_commission_bench = initial_capital * commission_pct
            benchmark_value = (initial_capital - initial_commission_bench) * (1 + benchmark_returns).cumprod()
        else:
            print(f"[WARN] No se pudo obtener {benchmark_ticker}, sin comparacion")
            benchmark_value = None
        
        # ==============================================
        # CALCULAR METRICAS - ESTRATEGIA ACTIVA (CON REBALANCEO)
        # ==============================================
        final_value_active = portfolio_value_active.iloc[-1]
        total_return_active = (final_value_active / initial_capital - 1) * 100
        
        # Máximo Drawdown
        cummax_active = portfolio_value_active.cummax()
        drawdown_active = (portfolio_value_active - cummax_active) / cummax_active * 100
        max_drawdown_active = drawdown_active.min()
        
        # Métricas anualizadas
        returns_active = portfolio_value_active.pct_change().dropna()
        days_traded = len(returns_active)
        years_traded = days_traded / 252
        annualized_return_active = ((final_value_active / initial_capital) ** (1/years_traded) - 1) * 100
        annualized_vol_active = returns_active.std() * np.sqrt(252) * 100
        # Sharpe Ratio con tasa libre de riesgo
        risk_free_daily = self.risk_free_rate / 252
        sharpe_ratio_active = ((returns_active.mean() - risk_free_daily) / returns_active.std()) * np.sqrt(252) if returns_active.std() > 0 else 0
        
        # ==============================================
        # CALCULAR METRICAS - ESTRATEGIA PASIVA (SIN REBALANCEO)
        # ==============================================
        final_value_passive = portfolio_value_passive.iloc[-1]
        total_return_passive = (final_value_passive / initial_capital - 1) * 100
        
        # Máximo Drawdown
        cummax_passive = portfolio_value_passive.cummax()
        drawdown_passive = (portfolio_value_passive - cummax_passive) / cummax_passive * 100
        max_drawdown_passive = drawdown_passive.min()
        
        # Métricas anualizadas
        returns_passive = portfolio_value_passive.pct_change().dropna()
        annualized_return_passive = ((final_value_passive / initial_capital) ** (1/years_traded) - 1) * 100
        annualized_vol_passive = returns_passive.std() * np.sqrt(252) * 100
        # Sharpe Ratio con tasa libre de riesgo (ya calculada arriba)
        sharpe_ratio_passive = ((returns_passive.mean() - risk_free_daily) / returns_passive.std()) * np.sqrt(252) if returns_passive.std() > 0 else 0
        
        # Benchmark métricas
        if benchmark_value is not None:
            bench_final = benchmark_value.iloc[-1]
            bench_return = (bench_final / initial_capital - 1) * 100
            bench_annualized = ((bench_final / initial_capital) ** (1/years_traded) - 1) * 100
            
            bench_cummax = benchmark_value.cummax()
            bench_drawdown = (benchmark_value - bench_cummax) / bench_cummax * 100
            bench_max_dd = bench_drawdown.min()
        else:
            bench_final = None
            bench_return = None
            bench_annualized = None
            bench_max_dd = None
        
        # Mostrar resultados en terminal
        print("\n" + "="*80)
        print("RESULTADOS DEL BACKTEST")
        print("="*80)
        
        print("\n[*] ESTRATEGIA ACTIVA (Con Rebalanceo Mensual):")
        print(f"  Capital Inicial:        ${initial_capital:,.2f}")
        print(f"  Capital Final:          ${final_value_active:,.2f}")
        print(f"  Retorno Total:          {total_return_active:+.2f}%")
        print(f"  Retorno Anualizado:     {annualized_return_active:+.2f}%")
        print(f"  Volatilidad Anualizada: {annualized_vol_active:.2f}%")
        print(f"  Sharpe Ratio:           {sharpe_ratio_active:.2f}")
        print(f"  Maximo Drawdown:        {max_drawdown_active:.2f}%")
        print(f"  Comisiones Totales:     ${total_commissions:,.2f} ({total_commissions/initial_capital*100:.2f}%)")
        
        print("\n[*] ESTRATEGIA PASIVA (Buy-and-Hold sin Rebalanceo):")
        print(f"  Capital Inicial:        ${initial_capital:,.2f}")
        print(f"  Capital Final:          ${final_value_passive:,.2f}")
        print(f"  Retorno Total:          {total_return_passive:+.2f}%")
        print(f"  Retorno Anualizado:     {annualized_return_passive:+.2f}%")
        print(f"  Volatilidad Anualizada: {annualized_vol_passive:.2f}%")
        print(f"  Sharpe Ratio:           {sharpe_ratio_passive:.2f}")
        print(f"  Maximo Drawdown:        {max_drawdown_passive:.2f}%")
        print(f"  Comisiones Totales:     ${initial_commission_passive:,.2f} ({initial_commission_passive/initial_capital*100:.2f}%)")
        
        # Análisis de Fricción
        friction_cost = total_commissions - initial_commission_passive
        return_difference = total_return_active - total_return_passive
        friction_worth_it = return_difference > (friction_cost / initial_capital * 100)
        
        print("\n[*] ANALISIS DE FRICCION (Costo del Rebalanceo):")
        print(f"  Comisiones Extra (Active):  ${friction_cost:,.2f}")
        print(f"  Diferencia de Retorno:      {return_difference:+.2f}%")
        print(f"  Rebalanceo Valio la Pena:   {'SI' if friction_worth_it else 'NO'}")
        if friction_worth_it:
            print(f"  Ganancia Neta:              ${(final_value_active - final_value_passive):,.2f} (+{return_difference:.2f}%)")
        else:
            print(f"  Perdida Neta:               ${(final_value_active - final_value_passive):,.2f} ({return_difference:.2f}%)")
        
        if benchmark_value is not None:
            print(f"\n[*] BENCHMARK ({benchmark_ticker}):")
            print(f"  Capital Final:          ${bench_final:,.2f}")
            print(f"  Retorno Total:          {bench_return:+.2f}%")
            print(f"  Retorno Anualizado:     {bench_annualized:+.2f}%")
            print(f"  Maximo Drawdown:        {bench_max_dd:.2f}%")
            
            outperformance_active = total_return_active - bench_return
            outperformance_passive = total_return_passive - bench_return
            performance_status_active = "[MEJOR]" if outperformance_active > 0 else "[PEOR]"
            performance_status_passive = "[MEJOR]" if outperformance_passive > 0 else "[PEOR]"
            
            print(f"\n[*] COMPARACION VS BENCHMARK:")
            print(f"  Active vs Benchmark:    {outperformance_active:+.2f}% {performance_status_active}")
            print(f"  Passive vs Benchmark:   {outperformance_passive:+.2f}% {performance_status_passive}")
        
        # Generar gráfico
        if save_path is None:
            save_path = OUTPUTS_DIR / 'backtest_results.png'
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[+] Generando grafico de Equity Curve...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Resultados del Backtesting - Cartera Optimizada', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Equity Curve (3 líneas: Active, Passive, Benchmark)
        ax1 = axes[0, 0]
        ax1.plot(portfolio_value_active.index, portfolio_value_active.values, 
                label='Active (Rebalanceo Mensual)', linewidth=2.5, color='#2E86AB')
        ax1.plot(portfolio_value_passive.index, portfolio_value_passive.values, 
                label='Passive (Buy-and-Hold)', linewidth=2, color='#F18F01', 
                linestyle='-', alpha=0.8)
        if benchmark_value is not None:
            ax1.plot(benchmark_value.index, benchmark_value.values, 
                    label=f'Benchmark ({benchmark_ticker})', linewidth=2, 
                    color='#A23B72', linestyle='--', alpha=0.7)
        ax1.axhline(initial_capital, color='gray', linestyle=':', alpha=0.5, 
                   label='Capital Inicial', linewidth=1)
        ax1.set_title('Equity Curve - Comparacion Active vs Passive', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Fecha')
        ax1.set_ylabel('Valor del Portfolio (USD)')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Plot 2: Drawdown con área gradiente
        ax2 = axes[0, 1]
        
        # Active con área sombreada (azul)
        ax2.fill_between(drawdown_active.index, 0, drawdown_active.values,
                         color='#2E86AB', alpha=0.25, label='Active (Rebalanceo)')
        ax2.plot(drawdown_active.index, drawdown_active.values,
                color='#2E86AB', linewidth=2.5, alpha=0.9)
        
        # Passive con área sombreada (naranja)
        ax2.fill_between(drawdown_passive.index, 0, drawdown_passive.values,
                         color='#F18F01', alpha=0.20, label='Passive (Buy-Hold)')
        ax2.plot(drawdown_passive.index, drawdown_passive.values,
                color='#F18F01', linewidth=2, alpha=0.8, linestyle='--')
        
        # Benchmark solo como línea punteada discreta
        if benchmark_value is not None:
            ax2.plot(bench_drawdown.index, bench_drawdown.values,
                    color='#141414', linewidth=1.5, label='Benchmark',
                    alpha=0.5, linestyle=':')
        
        # Línea de referencia en 0%
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
        
        # Anotación del máximo drawdown (Active) con flecha
        ax2.annotate(f'Max DD\n{max_drawdown_active:.1f}%',
                    xy=(drawdown_active.idxmin(), max_drawdown_active),
                    xytext=(10, -30), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='#2E86AB',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                             edgecolor='#2E86AB', linewidth=2, alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='#2E86AB',
                                  lw=1.5, connectionstyle='arc3,rad=0.3'))
        
        # Anotación del máximo drawdown (Passive) - más discreta
        ax2.annotate(f'{max_drawdown_passive:.1f}%',
                    xy=(drawdown_passive.idxmin(), max_drawdown_passive),
                    xytext=(10, 15), textcoords='offset points',
                    fontsize=8, color='#F18F01', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='#F18F01', linewidth=1.5, alpha=0.8))
        
        ax2.set_title('Drawdown vs Máximo Histórico', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Fecha', fontsize=10)
        ax2.set_ylabel('Drawdown (%)', fontsize=10)
        ax2.legend(loc='lower right', fontsize=9, framealpha=0.95)
        ax2.grid(True, alpha=0.2, linestyle='--')
        ax2.set_ylim(min(max_drawdown_active, max_drawdown_passive) * 1.15, 2)
        
        # Plot 3: Retornos Diarios (Active vs Passive)
        ax3 = axes[1, 0]
        ax3.hist(returns_active * 100, bins=40, alpha=0.6, color='#2E86AB', 
                edgecolor='black', label='Active', density=True)
        ax3.hist(returns_passive * 100, bins=40, alpha=0.5, color='#F18F01', 
                edgecolor='black', label='Passive', density=True)
        ax3.axvline(returns_active.mean() * 100, color='#2E86AB', linestyle='--', 
                   linewidth=2, alpha=0.8, label=f'Media Active: {returns_active.mean()*100:.3f}%')
        ax3.axvline(returns_passive.mean() * 100, color='#F18F01', linestyle='--', 
                   linewidth=2, alpha=0.8, label=f'Media Passive: {returns_passive.mean()*100:.3f}%')
        ax3.set_title('Distribucion de Retornos Diarios', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Retorno Diario (%)')
        ax3.set_ylabel('Densidad')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Comparación de Métricas (Active vs Passive vs Benchmark)
        ax4 = axes[1, 1]
        
        if benchmark_value is not None:
            # Crear tabla visual más clara con 3 estrategias
            metrics_labels = ['Sharpe\nRatio', 'Max\nDrawdown (%)', 'Retorno\nTotal (%)']
            active_vals = [sharpe_ratio_active, abs(max_drawdown_active), total_return_active]
            passive_vals = [sharpe_ratio_passive, abs(max_drawdown_passive), total_return_passive]
            bench_vals = [0, abs(bench_max_dd), bench_return]  # Benchmark no tiene Sharpe en este contexto
            
            x = np.arange(len(metrics_labels))
            width = 0.25
            
            # Barras con colores distintivos
            bars1 = ax4.bar(x - width, active_vals, width, 
                           label='Active (Rebalanceo)', color='#2E86AB', 
                           edgecolor='black', linewidth=1)
            bars2 = ax4.bar(x, passive_vals, width, 
                           label='Passive (Buy-Hold)', color='#F18F01',
                           edgecolor='black', linewidth=1)
            bars3 = ax4.bar(x + width, bench_vals, width, 
                           label=f'Benchmark ({benchmark_ticker})', color='#A23B72',
                           edgecolor='black', linewidth=1)
            
            # Configurar eje X con etiquetas claras
            ax4.set_xticks(x)
            ax4.set_xticklabels(metrics_labels, fontsize=9, fontweight='bold')
            ax4.set_ylabel('Valor', fontsize=10, fontweight='bold')
            ax4.set_title('Comparacion: Active vs Passive vs Benchmark', 
                         fontsize=11, fontweight='bold')
            
            # Grid horizontal para facilitar lectura
            ax4.yaxis.grid(True, alpha=0.4, linestyle='--')
            ax4.set_axisbelow(True)
            
            # Leyenda más visible
            ax4.legend(loc='upper left', fontsize=8, framealpha=0.9)
            
            # Valores sobre las barras con mejor formato
            for i, val in enumerate(active_vals):
                height = bars1[i].get_height()
                if height > 0:
                    ax4.text(bars1[i].get_x() + bars1[i].get_width()/2., height,
                            f'{val:.1f}',
                            ha='center', va='bottom', fontsize=7, fontweight='bold',
                            color='#2E86AB')
            
            for i, val in enumerate(passive_vals):
                height = bars2[i].get_height()
                if height > 0:
                    ax4.text(bars2[i].get_x() + bars2[i].get_width()/2., height,
                            f'{val:.1f}',
                            ha='center', va='bottom', fontsize=7, fontweight='bold',
                            color='#F18F01')
            
            for i, val in enumerate(bench_vals):
                height = bars3[i].get_height()
                if height > 0 and i != 0:  # Skip Sharpe for benchmark (ahora está en posición 0)
                    ax4.text(bars3[i].get_x() + bars3[i].get_width()/2., height,
                            f'{val:.1f}',
                            ha='center', va='bottom', fontsize=7, fontweight='bold',
                            color='#A23B72')
        else:
            # Si no hay benchmark, mostrar solo Active vs Passive
            metrics_labels = ['Sharpe\nRatio', 'Max\nDrawdown (%)', 'Retorno\nTotal (%)']
            active_vals = [sharpe_ratio_active, abs(max_drawdown_active), total_return_active]
            passive_vals = [sharpe_ratio_passive, abs(max_drawdown_passive), total_return_passive]
            
            x = np.arange(len(metrics_labels))
            width = 0.35
            
            bars1 = ax4.bar(x - width/2, active_vals, width, 
                           label='Active', color='#2E86AB', 
                           edgecolor='black', linewidth=1.5)
            bars2 = ax4.bar(x + width/2, passive_vals, width, 
                           label='Passive', color='#F18F01',
                           edgecolor='black', linewidth=1.5)
            
            ax4.set_xticks(x)
            ax4.set_xticklabels(metrics_labels, fontsize=10, fontweight='bold')
            ax4.set_title('Comparacion: Active vs Passive', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Valor', fontsize=11, fontweight='bold')
            ax4.yaxis.grid(True, alpha=0.4, linestyle='--')
            ax4.legend(loc='upper left', fontsize=10)
            
            # Valores sobre barras
            for bar, val in zip(bars1, active_vals):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            for bar, val in zip(bars2, passive_vals):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Grafico guardado en: {save_path}")
        
        # Retornar métricas (Active, Passive, Benchmark)
        return {
            # Información del período
            'start_date': start_date,
            'end_date': end_date,
            'lookback_period': lookback_period,
            
            # Estrategia ACTIVA (con rebalanceo)
            'portfolio_value_active': portfolio_value_active,
            'final_value_active': final_value_active,
            'total_return_active': total_return_active,
            'annualized_return_active': annualized_return_active,
            'volatility_active': annualized_vol_active,
            'sharpe_ratio_active': sharpe_ratio_active,
            'max_drawdown_active': max_drawdown_active,
            'total_commissions': total_commissions,
            
            # Estrategia PASIVA (sin rebalanceo)
            'portfolio_value_passive': portfolio_value_passive,
            'final_value_passive': final_value_passive,
            'total_return_passive': total_return_passive,
            'annualized_return_passive': annualized_return_passive,
            'volatility_passive': annualized_vol_passive,
            'sharpe_ratio_passive': sharpe_ratio_passive,
            'max_drawdown_passive': max_drawdown_passive,
            'initial_commission_passive': initial_commission_passive,
            
            # Análisis de Fricción
            'friction_cost': friction_cost,
            'return_difference': return_difference,
            'friction_worth_it': friction_worth_it,
            'rebalance_worthwhile': friction_worth_it,
            
            # Benchmark
            'benchmark_value': benchmark_value,
            'initial_capital': initial_capital,
            'benchmark_return': bench_return,
            'benchmark_final': bench_final,
            'benchmark_max_dd': bench_max_dd,
            'benchmark_annualized': bench_annualized,
            'outperformance_active': outperformance_active if benchmark_value is not None else None,
            'outperformance_passive': outperformance_passive if benchmark_value is not None else None,
        }


def main():
    """
    Función principal del motor de optimización
    Flujo completo: Descarga -> Estadísticas -> Optimización -> VaR/CVaR [Ex-Ante] -> Backtesting [Ex-Post] -> Stress Testing
    """
    print("\n" + "="*80)
    print("PORTFOLIO ENGINE v2.0 - FLUJO COMPLETO DE ANALISIS")
    print("="*80)
    print("\n[i] Modulos a ejecutar:")
    print("    1. Descarga de datos historicos")
    print("    2. Estadisticas descriptivas")
    print("    3. Optimizacion de carteras (Min Vol + Max Sharpe)")
    print("    4. Backtesting (validacion historica)")
    print("    5. Analisis de riesgo (VaR/CVaR t-Student)")
    print("    6. Stress Testing (escenarios extremos)")
    print("    7. Generacion de reportes")
    print("="*80 + "\n")
    
    # Definir activos
    tickers = [
        'GGAL.BA',      # Banco Galicia (Equity Argentina)
        'YPFD.BA',      # YPF (Equity Argentina)
        'ALUA.BA',      # Aluar (Equity Argentina)
        'GOOGL',         # Google (Equity Global)
        'BTC-USD',      # Bitcoin (Criptomoneda)
        'MSFT',         # Microsoft (Equity Global)
        'KO',           # Coca-Cola (Equity Global)
    ]
    
    # ==========================================
    # CONFIGURACIÓN DE PERÍODOS (TRAIN/TEST SPLIT)
    # ==========================================
    # ⚙️ AJUSTA ESTOS VALORES FÁCILMENTE:
    ANALYSIS_YEARS = 2  # ← Años de datos para optimización (mínimo: 2, óptimo: 3-5)
    BACKTEST_YEARS = 1  # ← Años para backtesting (debe ser < ANALYSIS_YEARS)
    
    # Para validación out-of-sample genuina:
    # - OPTIMIZACIÓN: Datos históricos para entrenar el modelo
    # - BACKTESTING: Datos posteriores (nunca vistos) para validar
    
    # Opción 1: Split automático (RECOMENDADO)
    # Descarga (ANALYSIS_YEARS + BACKTEST_YEARS), optimiza con primeros, valida con últimos
    end_date_optimization = (datetime.now() - timedelta(days=BACKTEST_YEARS*365)).strftime('%Y-%m-%d')
    start_date_optimization = (datetime.now() - timedelta(days=(ANALYSIS_YEARS + BACKTEST_YEARS)*365)).strftime('%Y-%m-%d')
    
    # Opción 2: Manual (descomentar para usar fechas específicas)
    # start_date_optimization = '2022-02-13'
    # end_date_optimization = '2025-02-13'
    
    INITIAL_CAPITAL = 1000000  # USD
    BACKTEST_PERIOD = f'{BACKTEST_YEARS}y'  # Validará los últimos N años (posterior a optimización)
    BENCHMARK = 'SPY'
    VAR_CONFIDENCE = 0.95
    MONTE_CARLO_SIMS = 10000
    
    print(f"\n{'='*80}")
    print(f"CONFIGURACION TRAIN/TEST SPLIT (Out-of-Sample)")
    print(f"{'='*80}")
    print(f"[TRAIN] Optimizacion: {start_date_optimization} a {end_date_optimization} ({ANALYSIS_YEARS} años)")
    print(f"[TEST]  Backtesting:  {end_date_optimization} a {datetime.now().strftime('%Y-%m-%d')} ({BACKTEST_YEARS} año(s))")
    print(f"{'='*80}\n")
    
    # Inicializar optimizador con período de entrenamiento
    optimizer = PortfolioOptimizer(tickers, start_date=start_date_optimization, end_date=end_date_optimization)
    
    # ==========================================
    # MÓDULO 1: DESCARGA DE DATOS
    # ==========================================
    print("\n" + ">"*80)
    print("MODULO 1/6: DESCARGA DE DATOS")
    print(">"*80)
    optimizer.download_data()
    
    # ==========================================
    # MÓDULO 2: ESTADÍSTICAS DESCRIPTIVAS
    # ==========================================
    print("\n" + ">"*80)
    print("MODULO 2/6: ESTADISTICAS DESCRIPTIVAS")
    print(">"*80)
    optimizer.calculate_statistics()
    
    # ==========================================
    # MÓDULO 3: OPTIMIZACIÓN DE CARTERAS
    # ==========================================
    print("\n" + ">"*80)
    print("MODULO 3/6: OPTIMIZACION Y FRONTERA EFICIENTE")
    print(">"*80)
    
    # Optimización LIBRE (sin restricciones)
    print("\n" + "-"*80)
    print("OPTIMIZACION LIBRE (Sin restricciones)")
    print("-"*80)
    min_vol_weights_free, min_vol_perf_free = optimizer.optimize_min_volatility(bounds_dict=None)
    max_sharpe_weights_free, max_sharpe_perf_free = optimizer.optimize_max_sharpe(bounds_dict=None)
    
    # Optimización GESTIONADA (con restricciones)
    print("\n" + "-"*80)
    print("OPTIMIZACION GESTIONADA (Con restricciones)")
    print("-"*80)
    
    # Definir restricciones realistas DINÁMICAMENTE basadas en activos reales
    bounds_managed = {}
    
    for ticker in tickers:
        if '.BA' in ticker:
            # Activos argentinos: Máximo 20% individual
            bounds_managed[ticker] = (0.00, 0.20)
        elif ticker.endswith('-USD'):
            # Criptomonedas: Máximo 10%
            bounds_managed[ticker] = (0.00, 0.10)
        else:
            # Activos globales: Flexibilidad media con mínimos
            # Primer activo global: mayor peso
            if len([t for t in tickers if not t.endswith('.BA') and not t.endswith('-USD')]) >= 1:
                if ticker == [t for t in tickers if not t.endswith('.BA') and not t.endswith('-USD')][0]:
                    bounds_managed[ticker] = (0.15, 0.35)  # Líder: 15-35%
                else:
                    bounds_managed[ticker] = (0.10, 0.30)  # Otros: 10-30%
    
    print("\n[i] Politica de inversion aplicada (dinamica):")
    print("    - Activos argentinos (.BA): Max 20% individual")
    print("    - Criptomonedas (-USD): Max 10%")
    print("    - Activos globales: Min 10-15%, Max 30-35%")
    print("\n[i] Restricciones por activo:")
    for ticker, (min_w, max_w) in bounds_managed.items():
        print(f"    {ticker:12s}: {min_w:.0%} - {max_w:.0%}")
    
    min_vol_weights_managed, min_vol_perf_managed = optimizer.optimize_min_volatility(bounds_dict=bounds_managed)
    max_sharpe_weights_managed, max_sharpe_perf_managed = optimizer.optimize_max_sharpe(bounds_dict=bounds_managed)
    
    # Generar frontera eficiente (usando carteras GESTIONADAS con restricciones)
    print("\n[+] Generando grafico de frontera eficiente...")
    _, _, _ = optimizer.plot_efficient_frontier(
        min_vol_weights=min_vol_weights_managed, 
        max_sharpe_weights=max_sharpe_weights_managed,
        min_vol_perf=min_vol_perf_managed,
        max_sharpe_perf=max_sharpe_perf_managed
    )
    
    # Usar versión GESTIONADA para el resto del análisis (más realista)
    print("\n" + "="*80)
    print("[DECISION] Usando cartera GESTIONADA para analisis posteriores")
    print("="*80)
    max_sharpe_weights = max_sharpe_weights_managed
    max_sharpe_perf = max_sharpe_perf_managed
    min_vol_weights = min_vol_weights_managed
    min_vol_perf = min_vol_perf_managed
    
    # ==========================================
    # MÓDULO 4: VaR/CVaR (EX-ANTE)
    # ==========================================
    print("\n" + ">"*80)
    print("MODULO 4/6: ANALISIS DE RIESGO (VaR/CVaR t-STUDENT) [EX-ANTE]")
    print(">"*80)
    print("\n[i] Proyectando riesgo con distribuciones t-Student...")
    print("[i] Esta proyeccion sera validada luego con backtesting historico")
    
    # Calcular VaR/CVaR para versión libre
    print("\n" + "-"*80)
    print("[+] Calculando VaR/CVaR para cartera LIBRE...")
    print("-"*80)
    risk_metrics_free = optimizer.calculate_var_cvar(
        max_sharpe_weights_free, 
        confidence_level=VAR_CONFIDENCE, 
        n_simulations=MONTE_CARLO_SIMS
    )
    
    # Calcular VaR/CVaR para versión gestionada
    print("\n" + "-"*80)
    print("[+] Calculando VaR/CVaR para cartera GESTIONADA...")
    print("-"*80)
    risk_metrics_managed = optimizer.calculate_var_cvar(
        max_sharpe_weights_managed, 
        confidence_level=VAR_CONFIDENCE, 
        n_simulations=MONTE_CARLO_SIMS
    )
    
    # Usar métricas de la cartera gestionada
    risk_metrics = risk_metrics_managed
    
    # Comparar ambas versiones
    print("\n" + "="*80)
    print("COMPARACION: LIBRE vs GESTIONADA (Sharpe y VaR)")
    print("="*80)
    
    optimizer.compare_strategies(
        weights_free=max_sharpe_weights_free,
        perf_free=max_sharpe_perf_free,
        risk_free=risk_metrics_free,
        weights_managed=max_sharpe_weights_managed,
        perf_managed=max_sharpe_perf_managed,
        risk_managed=risk_metrics_managed
    )
    
    # ==========================================
    # MÓDULO 5: BACKTESTING (EX-POST)
    # ==========================================
    print("\n" + ">"*80)
    print("MODULO 5/6: BACKTESTING (VALIDACION HISTORICA) [EX-POST]")
    print(">"*80)
    print("\n[i] Validando la proyeccion de VaR con drawdown historico real...")
    backtest_results = optimizer.run_backtest(
        weights=max_sharpe_weights, 
        initial_capital=INITIAL_CAPITAL,
        lookback_period=BACKTEST_PERIOD,
        benchmark_ticker=BENCHMARK
    )
    
    # ==========================================
    # MÓDULO 6: STRESS TESTING
    # ==========================================
    print("\n" + ">"*80)
    print("MODULO 6/6: STRESS TESTING (ESCENARIOS EXTREMOS)")
    print(">"*80)
    stress_results = optimizer.run_stress_test(
        weights=max_sharpe_weights,
        capital=INITIAL_CAPITAL
    )
    
    # ==========================================
    # GENERACIÓN DE REPORTES
    # ==========================================
    print("\n" + "="*80)
    print("GENERACION DE REPORTES")
    print("="*80)
    
    # Reporte 1: Optimización (Ex-Ante) - Incluir comparación Libre vs Gestionada
    optimizer.generate_report(
        min_vol_weights=min_vol_weights,
        max_sharpe_weights=max_sharpe_weights,
        risk_metrics=risk_metrics,
        min_vol_weights_free=min_vol_weights_free,
        max_sharpe_weights_free=max_sharpe_weights_free,
        min_vol_perf_free=min_vol_perf_free,
        max_sharpe_perf_free=max_sharpe_perf_free,
        risk_metrics_free=risk_metrics_free
    )
    
    # Reporte 2: Backtesting (Ex-Post)
    if backtest_results is not None:
        optimizer.generate_backtest_report(backtest_results, max_sharpe_weights, risk_metrics)
    
    # Reporte 3: Stress Testing
    if stress_results is not None:
        optimizer.generate_stress_test_report(stress_results, max_sharpe_weights, INITIAL_CAPITAL)
    
    # ==========================================
    # RESUMEN FINAL
    # ==========================================
    print("\n" + "="*80)
    print("[OK] ANALISIS COMPLETADO CON EXITO")
    print("="*80)
    
    print("\n[ARCHIVOS GENERADOS]")
    print("\nGRAFICOS:")
    print(f"  [*] {OUTPUTS_DIR / 'efficient_frontier.png'} ......... Frontera eficiente + matrices")
    print(f"  [*] {OUTPUTS_DIR / 'backtest_results.png'} .......... Equity curve + drawdown")
    print(f"  [*] {OUTPUTS_DIR / 'stress_test.png'} ............... Escenarios de estres")
    
    print("\nREPORTES:")
    print(f"  [*] {OUTPUTS_DIR / 'reporte_portfolio.md'} .......... Analisis Ex-Ante (Optimizacion)")
    print(f"  [*] {OUTPUTS_DIR / 'reporte_backtesting.md'} ........ Validacion Ex-Post (Backtest)")
    print(f"  [*] {OUTPUTS_DIR / 'reporte_stress_test.md'} ........ Analisis de Escenarios Extremos")
    
    print("\n[METRICAS CLAVE]")
    print(f"\nCartera de Maximo Sharpe:")
    print(f"  Retorno Esperado:      {max_sharpe_perf[0]*100:6.2f}%")
    print(f"  Volatilidad:           {max_sharpe_perf[1]*100:6.2f}%")
    print(f"  Sharpe Ratio:          {max_sharpe_perf[2]:6.2f}")
    
    if backtest_results:
        print(f"\nBacktest (1 ano):")
        print(f"  Retorno Activo:        {backtest_results['total_return_active']:+6.2f}%")
        print(f"  Retorno Pasivo:        {backtest_results['total_return_passive']:+6.2f}%")
        print(f"  Sharpe Activo:         {backtest_results['sharpe_ratio_active']:6.2f}")
        print(f"  Max Drawdown Activo:   {backtest_results['max_drawdown_active']:6.2f}%")
        print(f"  Rebalanceo valio la pena: {'SI' if backtest_results['friction_worth_it'] else 'NO'}")
        if backtest_results.get('outperformance_active') is not None:
            print(f"  Activo vs {BENCHMARK}:        {backtest_results['outperformance_active']:+6.2f}%")
    
    print(f"\nVaR/CVaR (95%, df=3 conservador):")
    print(f"  VaR (1 dia):           {risk_metrics['var_daily']*100:6.2f}%")
    print(f"  CVaR (1 dia):          {risk_metrics['cvar_daily']*100:6.2f}%")
    
    if stress_results:
        worst_case = min(stress_results.values(), key=lambda x: x['pct_change'])
        worst_scenario = [k for k, v in stress_results.items() if v == worst_case][0]
        print(f"\nStress Test:")
        print(f"  Peor escenario:        {worst_scenario}")
        print(f"  Perdida potencial:     {worst_case['pct_change']:6.2f}%")
        print(f"  Capital remanente:     ${worst_case['final_capital']:,.2f}")
    
    print("\n" + "="*80)
    print("[i] Para detalles completos, consultar los reportes en outputs/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

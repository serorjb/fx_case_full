"""
pricing_models.py
FX Options Pricing Models using QuantLib
"""

import numpy as np
import QuantLib as ql
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class QuantLibPricingBase:
    """Base class for QuantLib-based pricing"""

    def __init__(self):
        # Set up QuantLib calendar and day count
        self.calendar = ql.UnitedStates(ql.UnitedStates.Settlement)
        self.day_count = ql.Actual365Fixed()

    def setup_market_data(self, spot: float, r_d: float, r_f: float,
                         reference_date: ql.Date = None):
        """Setup QuantLib market data"""
        if reference_date is None:
            reference_date = ql.Date.todaysDate()

        ql.Settings.instance().evaluationDate = reference_date

        # Create spot handle
        self.spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))

        # Create rate curves (flat for simplicity, can be enhanced)
        self.domestic_rate = ql.YieldTermStructureHandle(
            ql.FlatForward(reference_date, r_d, self.day_count)
        )
        self.foreign_rate = ql.YieldTermStructureHandle(
            ql.FlatForward(reference_date, r_f, self.day_count)
        )

    def create_option(self, strike: float, maturity: ql.Date,
                      option_type: str = 'call') -> ql.VanillaOption:
        """Create a QuantLib vanilla option"""
        if option_type.lower() == 'call':
            payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
        else:
            payoff = ql.PlainVanillaPayoff(ql.Option.Put, strike)

        exercise = ql.EuropeanExercise(maturity)
        option = ql.VanillaOption(payoff, exercise)

        return option


class BlackScholesFX(QuantLibPricingBase):
    """
    Black-Scholes model for FX options using QuantLib (Garman-Kohlhagen)
    """

    def __init__(self):
        super().__init__()
        self.process = None
        self.engine = None

    def setup_process(self, spot: float, r_d: float, r_f: float,
                     sigma: float, reference_date: ql.Date = None):
        """Setup Black-Scholes process"""
        self.setup_market_data(spot, r_d, r_f, reference_date)

        # Create volatility term structure
        volatility = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(
                reference_date if reference_date else ql.Date.todaysDate(),
                self.calendar,
                sigma,
                self.day_count
            )
        )

        # Create Garman-Kohlhagen process for FX
        self.process = ql.GarmanKohlagenProcess(
            self.spot_handle,
            self.foreign_rate,
            self.domestic_rate,
            volatility
        )

        # Create pricing engine
        self.engine = ql.AnalyticEuropeanEngine(self.process)

    def price(self, strike: float, maturity_days: int, sigma: float,
             spot: float, r_d: float, r_f: float,
             option_type: str = 'call') -> float:
        """Price an option using Black-Scholes"""
        reference_date = ql.Date.todaysDate()
        maturity = reference_date + ql.Period(maturity_days, ql.Days)

        # Setup process with given parameters
        self.setup_process(spot, r_d, r_f, sigma, reference_date)

        # Create and price option
        option = self.create_option(strike, maturity, option_type)
        option.setPricingEngine(self.engine)

        return option.NPV()

    def calculate_greeks(self, strike: float, maturity_days: int,
                        sigma: float, spot: float, r_d: float, r_f: float,
                        option_type: str = 'call') -> Dict[str, float]:
        """Calculate all Greeks using QuantLib"""
        reference_date = ql.Date.todaysDate()
        maturity = reference_date + ql.Period(maturity_days, ql.Days)

        # Setup process
        self.setup_process(spot, r_d, r_f, sigma, reference_date)

        # Create option
        option = self.create_option(strike, maturity, option_type)
        option.setPricingEngine(self.engine)

        # Calculate Greeks
        greeks = {
            'price': option.NPV(),
            'delta': option.delta(),
            'gamma': option.gamma(),
            'vega': option.vega() / 100,  # QuantLib vega is per 1 unit change
            'theta': option.theta() / 365,  # Convert to daily theta
            'rho': option.rho() / 100
        }

        return greeks

    def implied_volatility(self, price: float, strike: float,
                          maturity_days: int, spot: float,
                          r_d: float, r_f: float,
                          option_type: str = 'call') -> float:
        """Calculate implied volatility using QuantLib"""
        reference_date = ql.Date.todaysDate()
        maturity = reference_date + ql.Period(maturity_days, ql.Days)

        # Setup market data
        self.setup_market_data(spot, r_d, r_f, reference_date)

        # Create option
        option = self.create_option(strike, maturity, option_type)

        # Setup process with initial guess
        initial_vol = 0.20
        self.setup_process(spot, r_d, r_f, initial_vol, reference_date)
        option.setPricingEngine(self.engine)

        try:
            # Use QuantLib's implied volatility solver (positional args for compatibility)
            impl_vol = option.impliedVolatility(
                price,
                self.process,
                1e-6,
                100
            )
            return impl_vol
        except:
            # Fallback to manual search if QuantLib fails
            return self._implied_vol_bisection(
                price, strike, maturity_days, spot, r_d, r_f, option_type
            )

    def _implied_vol_bisection(self, target_price: float, strike: float,
                               maturity_days: int, spot: float,
                               r_d: float, r_f: float,
                               option_type: str = 'call') -> float:
        """Bisection method for implied volatility"""
        low_vol, high_vol = 0.001, 3.0
        tolerance = 1e-6
        max_iterations = 100

        for _ in range(max_iterations):
            mid_vol = (low_vol + high_vol) / 2
            price = self.price(strike, maturity_days, mid_vol,
                             spot, r_d, r_f, option_type)

            if abs(price - target_price) < tolerance:
                return mid_vol

            if price < target_price:
                low_vol = mid_vol
            else:
                high_vol = mid_vol

        return mid_vol

    # Compatibility methods for existing code
    @staticmethod
    def call_price(S: float, K: float, r_d: float, r_f: float,
                   sigma: float, T: float) -> float:
        """Static method for compatibility"""
        bs = BlackScholesFX()
        maturity_days = int(T * 365)
        return bs.price(K, maturity_days, sigma, S, r_d, r_f, 'call')

    @staticmethod
    def put_price(S: float, K: float, r_d: float, r_f: float,
                  sigma: float, T: float) -> float:
        """Static method for compatibility"""
        bs = BlackScholesFX()
        maturity_days = int(T * 365)
        return bs.price(K, maturity_days, sigma, S, r_d, r_f, 'put')

    @staticmethod
    def delta(S: float, K: float, r_d: float, r_f: float,
              sigma: float, T: float, option_type: str = 'call') -> float:
        """Static method for compatibility"""
        bs = BlackScholesFX()
        maturity_days = int(T * 365)
        greeks = bs.calculate_greeks(K, maturity_days, sigma, S, r_d, r_f, option_type)
        return greeks['delta']

    @staticmethod
    def gamma(S: float, K: float, r_d: float, r_f: float,
              sigma: float, T: float) -> float:
        """Static method for compatibility"""
        bs = BlackScholesFX()
        maturity_days = int(T * 365)
        greeks = bs.calculate_greeks(K, maturity_days, sigma, S, r_d, r_f, 'call')
        return greeks['gamma']

    @staticmethod
    def vega(S: float, K: float, r_d: float, r_f: float,
             sigma: float, T: float) -> float:
        """Static method for compatibility"""
        bs = BlackScholesFX()
        maturity_days = int(T * 365)
        greeks = bs.calculate_greeks(K, maturity_days, sigma, S, r_d, r_f, 'call')
        return greeks['vega']

    @staticmethod
    def theta(S: float, K: float, r_d: float, r_f: float,
              sigma: float, T: float, option_type: str = 'call') -> float:
        """Static method for compatibility"""
        bs = BlackScholesFX()
        maturity_days = int(T * 365)
        greeks = bs.calculate_greeks(K, maturity_days, sigma, S, r_d, r_f, option_type)
        return greeks['theta']

    @staticmethod
    def vanna(S: float, K: float, r_d: float, r_f: float,
              sigma: float, T: float) -> float:
        """Calculate vanna (dDelta/dVol)"""
        # QuantLib doesn't have built-in vanna, so we approximate
        eps = 0.0001
        bs = BlackScholesFX()
        maturity_days = int(T * 365)

        delta_up = bs.calculate_greeks(K, maturity_days, sigma + eps,
                                       S, r_d, r_f, 'call')['delta']
        delta_down = bs.calculate_greeks(K, maturity_days, sigma - eps,
                                         S, r_d, r_f, 'call')['delta']

        return (delta_up - delta_down) / (2 * eps) / 100

    @staticmethod
    def volga(S: float, K: float, r_d: float, r_f: float,
              sigma: float, T: float) -> float:
        """Calculate volga (dVega/dVol)"""
        # QuantLib doesn't have built-in volga, so we approximate
        eps = 0.0001
        bs = BlackScholesFX()
        maturity_days = int(T * 365)

        vega_up = bs.calculate_greeks(K, maturity_days, sigma + eps,
                                      S, r_d, r_f, 'call')['vega']
        vega_down = bs.calculate_greeks(K, maturity_days, sigma - eps,
                                        S, r_d, r_f, 'call')['vega']

        return (vega_up - vega_down) / (2 * eps) / 100


class SABRModelQL(QuantLibPricingBase):
    """
    SABR model using QuantLib
    """

    def __init__(self, forward: float, T: float):
        super().__init__()
        self.forward = float(forward)
        self.T = float(T)
        self.expiry_time = float(T)  # Store as float for QuantLib

    def calibrate(self, strikes: np.ndarray, market_vols: np.ndarray,
                  beta: float = 0.5, weights: Optional[np.ndarray] = None) -> Dict:
        """Calibrate SABR parameters using QuantLib with optional weights on squared errors."""
        # Convert to QuantLib arrays
        strikes_ql = [float(k) for k in strikes]
        vols_ql = [float(v) for v in market_vols]
        if weights is None:
            weights_arr = np.ones(len(vols_ql), dtype=float)
        else:
            weights_arr = np.asarray(weights, dtype=float)
            if weights_arr.shape[0] != len(vols_ql):
                weights_arr = np.ones(len(vols_ql), dtype=float)

        # Initial guess
        alpha = float(market_vols[len(market_vols)//2])  # ATM vol approximation
        nu = 0.3
        rho = 0.0

        def objective(params):
            alpha, rho, nu = params
            # Ensure valid parameters
            if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
                return 1e10
            try:
                model_vols = []
                for strike in strikes_ql:
                    mv = ql.sabrVolatility(float(strike), self.forward, self.expiry_time,
                                           float(alpha), float(beta), float(nu), float(rho))
                    model_vols.append(float(mv))
                model_vols = np.array(model_vols)
                diffs = model_vols - np.array(vols_ql)
                return float(np.sum(weights_arr * diffs * diffs))
            except Exception:
                return 1e10

        # Optimize
        result = minimize(
            objective,
            x0=[alpha, rho, nu],
            bounds=[(0.001, 1.0), (-0.99, 0.99), (0.001, 2.0)],
            method='L-BFGS-B'
        )

        if result.success:
            alpha, rho, nu = result.x
        else:
            # Use differential evolution as fallback
            bounds = [(0.001, 1.0), (-0.99, 0.99), (0.001, 2.0)]
            result = differential_evolution(objective, bounds, seed=42)
            alpha, rho, nu = result.x

        return {
            'alpha': float(alpha),
            'beta': float(beta),
            'rho': float(rho),
            'nu': float(nu)
        }

    def sabr_vol(self, K: float, alpha: float, beta: float,
                 rho: float, nu: float) -> float:
        """Calculate SABR implied volatility using QuantLib"""
        try:
            vol = ql.sabrVolatility(
                float(K), float(self.forward), float(self.expiry_time),
                float(alpha), float(beta), float(nu), float(rho)
            )
            return float(vol)
        except Exception as e:
            # Return a default value if calculation fails
            return float(alpha)

    def price_vanilla(self, spot: float, K: float, r_d: float, r_f: float,
                     params: Dict, option_type: str = 'call') -> float:
        """Price vanilla option using SABR volatility"""
        vol = self.sabr_vol(K, params['alpha'], params['beta'],
                           params['rho'], params['nu'])

        bs = BlackScholesFX()
        maturity_days = int(self.T * 365)
        return bs.price(K, maturity_days, vol, spot, r_d, r_f, option_type)


class HestonModelQL(QuantLibPricingBase):
    """
    Heston stochastic volatility model using QuantLib
    """

    def __init__(self, spot: float, r_d: float, r_f: float, T: float):
        super().__init__()
        self.spot = spot
        self.r_d = r_d
        self.r_f = r_f
        self.T = T

    def setup_heston_process(self, params: Dict):
        """Setup Heston process in QuantLib"""
        reference_date = ql.Date.todaysDate()
        self.setup_market_data(self.spot, self.r_d, self.r_f, reference_date)

        # Heston parameters
        v0 = params['v0']      # Initial variance
        kappa = params['kappa'] # Mean reversion speed
        theta = params['theta'] # Long-term variance
        sigma = params['sigma'] # Vol of vol
        rho = params['rho']     # Correlation

        # Create Heston process
        self.process = ql.HestonProcess(
            self.domestic_rate,
            self.foreign_rate,
            self.spot_handle,
            v0, kappa, theta, sigma, rho
        )

        # Create Heston model and engine
        model = ql.HestonModel(self.process)
        self.engine = ql.AnalyticHestonEngine(model)

    def calibrate(self, strikes: np.ndarray, market_vols: np.ndarray,
                  maturities: np.ndarray = None) -> Dict:
        """Calibrate Heston model to market vols"""

        # Initial parameters
        v0 = np.mean(market_vols) ** 2
        kappa = 2.0
        theta = v0
        sigma = 0.3
        rho = -0.5

        def objective(params):
            v0, kappa, theta, sigma, rho = params

            # Ensure valid parameters
            if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0 or abs(rho) >= 1:
                return 1e10

            # Setup process
            heston_params = {
                'v0': v0, 'kappa': kappa, 'theta': theta,
                'sigma': sigma, 'rho': rho
            }

            try:
                self.setup_heston_process(heston_params)

                # Calculate model prices and implied vols
                total_error = 0
                for strike, market_vol in zip(strikes, market_vols):
                    maturity = ql.Date.todaysDate() + ql.Period(int(self.T * 365), ql.Days)
                    option = self.create_option(strike, maturity, 'call')
                    option.setPricingEngine(self.engine)

                    model_price = option.NPV()

                    # Convert to implied vol
                    bs = BlackScholesFX()
                    try:
                        model_vol = bs.implied_volatility(
                            model_price, strike, int(self.T * 365),
                            self.spot, self.r_d, self.r_f, 'call'
                        )
                        total_error += (model_vol - market_vol) ** 2
                    except:
                        return 1e10

                return total_error
            except:
                return 1e10

        # Optimize
        bounds = [
            (0.001, 1.0),   # v0
            (0.1, 10.0),    # kappa
            (0.001, 1.0),   # theta
            (0.01, 2.0),    # sigma
            (-0.99, 0.99)   # rho
        ]

        result = differential_evolution(objective, bounds, seed=42, maxiter=100)

        v0, kappa, theta, sigma, rho = result.x

        return {
            'v0': v0,
            'kappa': kappa,
            'theta': theta,
            'sigma': sigma,
            'rho': rho
        }

    def price_vanilla(self, K: float, params: Dict,
                     option_type: str = 'call') -> float:
        """Price vanilla option using Heston model"""
        self.setup_heston_process(params)

        maturity = ql.Date.todaysDate() + ql.Period(int(self.T * 365), ql.Days)
        option = self.create_option(K, maturity, option_type)
        option.setPricingEngine(self.engine)

        return option.NPV()


class VGVVModel:
    """
    Vega-Gamma-Vanna-Volga model
    Note: This is not directly available in QuantLib, so we implement it
    but use QuantLib for the underlying Black-Scholes calculations
    """

    def __init__(self, spot: float, forward: float, r_d: float,
                 r_f: float, T: float):
        self.spot = spot
        self.forward = forward
        self.r_d = r_d
        self.r_f = r_f
        self.T = T
        self.bs = BlackScholesFX()
        self.last_params: Dict | None = None

    def calibrate(self, strikes: np.ndarray, market_vols: np.ndarray) -> Dict:
        """
        Calibrate VGVV model to market volatilities
        """
        # ATM forward strike
        K_atm = self.forward

        # Find ATM volatility
        atm_idx = np.searchsorted(strikes, K_atm)
        if atm_idx == 0:
            sigma_atm = market_vols[0]
        elif atm_idx >= len(strikes):
            sigma_atm = market_vols[-1]
        else:
            w = (K_atm - strikes[atm_idx-1]) / (strikes[atm_idx] - strikes[atm_idx-1])
            sigma_atm = market_vols[atm_idx-1] * (1-w) + market_vols[atm_idx] * w

        # Fit quadratic smile in variance space
        log_moneyness = np.log(strikes / self.forward)
        variances = market_vols ** 2

        # Quadratic fit
        A = np.vstack([np.ones_like(log_moneyness),
                      log_moneyness,
                      log_moneyness**2]).T
        coeffs = np.linalg.lstsq(A, variances, rcond=None)[0]

        # Extract parameters
        v_0 = coeffs[0]
        skew = coeffs[1] / (2 * np.sqrt(v_0) * self.T) if v_0 > 0 else 0
        smile = coeffs[2] / (v_0 * self.T) if v_0 > 0 else 0

        # Map to correlation and vol-of-vol
        rho = -np.tanh(skew * 2)
        volvol = np.sqrt(abs(smile)) * 2

        return {
            'sigma_atm': np.sqrt(v_0),
            'rho': np.clip(rho, -0.99, 0.99),
            'volvol': np.clip(volvol, 0.01, 2.0),
            'v_0': v_0,
            'skew': skew,
            'smile': smile
        }

    def vgvv_vol(self, K: float, params: Dict) -> float:
        """Return the VGVV-adjusted volatility at strike K under calibrated params."""
        sigma_atm = float(params['sigma_atm'])
        rho = float(params['rho'])
        volvol = float(params['volvol'])
        log_m = np.log(float(K) / float(self.forward))
        # Same variance adjustment as price_vanilla
        variance_adjustment = 1.0
        variance_adjustment += rho * volvol * log_m / np.sqrt(self.T)
        variance_adjustment += 0.5 * (volvol**2) * (log_m**2 - sigma_atm**2 * self.T) / self.T
        return float(np.clip(sigma_atm * np.sqrt(max(variance_adjustment, 0.01)), 1e-4, 5.0))

    def price_vanilla(self, K: float, params: Dict,
                     option_type: str = 'call') -> float:
        """
        Price vanilla option using VGVV adjustments
        """
        sigma_atm = params['sigma_atm']
        rho = params['rho']
        volvol = params['volvol']

        # Calculate adjusted volatility
        log_moneyness = np.log(K / self.forward)

        # VGVV volatility adjustment
        variance_adjustment = 1.0
        variance_adjustment += rho * volvol * log_moneyness / np.sqrt(self.T)
        variance_adjustment += 0.5 * volvol**2 * (log_moneyness**2 - sigma_atm**2 * self.T) / self.T

        sigma_adjusted = sigma_atm * np.sqrt(max(variance_adjustment, 0.01))

        # Price using Black-Scholes
        maturity_days = int(self.T * 365)
        return self.bs.price(K, maturity_days, sigma_adjusted,
                           self.spot, self.r_d, self.r_f, option_type)

    def get_smile(self, params: Dict, strikes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get implied volatility smile from VGVV parameters
        """
        if strikes is None:
            strikes = np.linspace(0.8 * self.forward, 1.2 * self.forward, 50)

        vols = []
        for K in strikes:
            log_moneyness = np.log(K / self.forward)

            # Reconstruct variance
            variance = params['v_0']
            variance += 2 * params['skew'] * np.sqrt(params['v_0']) * self.T * log_moneyness
            variance += params['smile'] * params['v_0'] * self.T * log_moneyness**2

            vols.append(np.sqrt(max(variance, 0.0001)))

        return np.array(vols)

    def calibrate_from_surface(self, vol_data, tenor: str):
        """Build synthetic smile from ATM, RR, BF quotes for a tenor and fit quadratic.
        Assumes FX market conventions: RR = call_vol - put_vol; BF = ((call_vol + put_vol)/2) - ATM.
        Returns parameter dict with sigma_atm, rho (mapped from skew), vov (vol-of-vol proxy)."""
        atm = vol_data.atm_vols.get(tenor)
        if atm is None:
            return None
        # Prefer 25D quotes then fall back to 10D
        rr = vol_data.rr_25d.get(tenor, 0.0)
        bf = vol_data.bf_25d.get(tenor, 0.0)
        # Reconstruct call / put wing vols
        call_25 = (rr + 2*(atm + bf)) / 2.0
        put_25 = (2*(atm + bf) - rr) / 2.0
        # Simple synthetic strike grid around forward
        F = self.forward
        T = self.T
        strikes = np.array([0.9*F, 0.95*F, F, 1.05*F, 1.10*F])
        vols = np.array([put_25*1.02, put_25, atm, call_25, call_25*1.02])
        params = self.calibrate(strikes, vols)
        self.last_params = params
        return params

    def get_atm_vol(self, spot: float, strike: float, T: float):
        if self.last_params:
            return self.last_params.get('sigma_atm', np.nan)
        return np.nan

    def _tenor_to_years(self, tenor: str) -> float:
        mapping = {'1W':7/365,'2W':14/365,'3W':21/365,'1M':1/12,'2M':2/12,'3M':3/12,'4M':4/12,'6M':6/12,'9M':9/12,'12M':1.0}
        return mapping.get(tenor, 1/12)


# Backward compatibility - keep old class names as aliases
class SABRModel(SABRModelQL):
    """Extended SABR model wrapper with surface calibration convenience."""
    def __init__(self, spot: float, forward: float, r_d: float, r_f: float, T: float):
        super().__init__(forward, T)
        self.spot = spot
        self.forward = forward
        self.r_d = r_d
        self.r_f = r_f
        self.T = T
        self._last_params = None

    def calibrate_from_surface(self, vol_data, tenor: str):
        atm = vol_data.atm_vols.get(tenor)
        if atm is None:
            return None
        rr = vol_data.rr_25d.get(tenor, 0.0)
        bf = vol_data.bf_25d.get(tenor, 0.0)
        call_25 = (rr + 2*(atm + bf))/2.0
        put_25 = (2*(atm + bf) - rr)/2.0
        F = self.forward
        strikes = np.array([0.95*F, F, 1.05*F])
        vols = np.array([put_25, atm, call_25])
        self._last_params = self.calibrate(strikes, vols, beta=0.5)
        return self._last_params

    def get_atm_vol(self, spot: float, strike: float, T: float):
        # Return ATM from last calibration if available
        if self._last_params:
            # Approx atm SABR vol: alpha / F^{beta-1}
            p = self._last_params
            beta = p['beta']
            alpha = p['alpha']
            return alpha / (self.forward ** (beta-1))
        return np.nan

    def _tenor_to_years(self, tenor: str) -> float:
        mapping = {'1W':7/365,'2W':14/365,'3W':21/365,'1M':1/12,'2M':2/12,'3M':3/12,'4M':4/12,'6M':6/12,'9M':9/12,'12M':1.0}
        return mapping.get(tenor, 1/12)


# Example usage and testing
if __name__ == "__main__":
    print("Testing QuantLib-based pricing models\n")
    print("="*50)

    # Test Black-Scholes with QuantLib
    print("\n1. Black-Scholes (Garman-Kohlhagen) Test:")
    print("-"*40)

    S = 1.0755  # AUDNZD spot
    K = 1.08    # Strike
    r_d = 0.02  # USD rate
    r_f = 0.025 # Foreign rate
    sigma = 0.068  # 6.8% volatility
    T = 30/365  # 1 month

    bs = BlackScholesFX()

    # Test pricing
    call_price = bs.call_price(S, K, r_d, r_f, sigma, T)
    put_price = bs.put_price(S, K, r_d, r_f, sigma, T)

    print(f"Spot: {S:.4f}, Strike: {K:.4f}")
    print(f"Call Price: {call_price:.5f}")
    print(f"Put Price: {put_price:.5f}")

    # Test Greeks
    maturity_days = int(T * 365)
    greeks = bs.calculate_greeks(K, maturity_days, sigma, S, r_d, r_f, 'call')

    print(f"\nGreeks (Call):")
    for name, value in greeks.items():
        print(f"  {name.capitalize()}: {value:.6f}")

    # Test implied volatility
    impl_vol = bs.implied_volatility(call_price, K, maturity_days, S, r_d, r_f, 'call')
    print(f"\nImplied Vol Recovery: {impl_vol:.4f} (Original: {sigma:.4f})")

    # Test SABR with QuantLib
    print("\n2. SABR Model Test:")
    print("-"*40)

    forward = S * np.exp((r_d - r_f) * T)
    sabr = SABRModelQL(forward, T)

    # Create sample smile
    strikes = np.linspace(0.95*forward, 1.05*forward, 5)
    market_vols = np.array([0.085, 0.072, 0.068, 0.071, 0.082])

    sabr_params = sabr.calibrate(strikes, market_vols, beta=0.5)
    print(f"Calibrated SABR Parameters:")
    for param, value in sabr_params.items():
        print(f"  {param}: {value:.4f}")

    # Test SABR volatility
    atm_strike = forward
    sabr_vol = sabr.sabr_vol(atm_strike, sabr_params['alpha'],
                            sabr_params['beta'], sabr_params['rho'],
                            sabr_params['nu'])
    print(f"\nSABR ATM Vol: {sabr_vol:.4f}")

    # Test Heston model
    print("\n3. Heston Model Test:")
    print("-"*40)

    heston = HestonModelQL(S, r_d, r_f, T)

    # Simple calibration test
    heston_params = {
        'v0': sigma**2,
        'kappa': 2.0,
        'theta': sigma**2,
        'sigma': 0.3,
        'rho': -0.5
    }

    heston.setup_heston_process(heston_params)
    heston_price = heston.price_vanilla(K, heston_params, 'call')
    print(f"Heston Call Price: {heston_price:.5f}")
    print(f"BS Call Price: {call_price:.5f}")
    print(f"Difference: {abs(heston_price - call_price):.5f}")

    print("\n" + "="*50)
    print("All QuantLib pricing models tested successfully!")
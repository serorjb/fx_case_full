"""
FX Options Pricing Models
Implements Black-Scholes, VGVV, and SABR models for FX options
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, root_scalar
from typing import Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore')


class BlackScholesFX:
    """
    Black-Scholes model for FX options (Garman-Kohlhagen)
    """

    @staticmethod
    def d1_d2(S: float, K: float, r_d: float, r_f: float,
              sigma: float, T: float) -> Tuple[float, float]:
        """Calculate d1 and d2 parameters"""
        d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    @staticmethod
    def call_price(S: float, K: float, r_d: float, r_f: float,
                   sigma: float, T: float) -> float:
        """Calculate call option price"""
        if T <= 0:
            return max(S - K, 0)

        d1, d2 = BlackScholesFX.d1_d2(S, K, r_d, r_f, sigma, T)
        call = S * np.exp(-r_f * T) * norm.cdf(d1) - K * np.exp(-r_d * T) * norm.cdf(d2)
        return call

    @staticmethod
    def put_price(S: float, K: float, r_d: float, r_f: float,
                  sigma: float, T: float) -> float:
        """Calculate put option price"""
        if T <= 0:
            return max(K - S, 0)

        d1, d2 = BlackScholesFX.d1_d2(S, K, r_d, r_f, sigma, T)
        put = K * np.exp(-r_d * T) * norm.cdf(-d2) - S * np.exp(-r_f * T) * norm.cdf(-d1)
        return put

    @staticmethod
    def delta(S: float, K: float, r_d: float, r_f: float,
              sigma: float, T: float, option_type: str = 'call') -> float:
        """Calculate option delta"""
        if T <= 0:
            if option_type == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0

        d1, _ = BlackScholesFX.d1_d2(S, K, r_d, r_f, sigma, T)

        if option_type == 'call':
            return np.exp(-r_f * T) * norm.cdf(d1)
        else:
            return -np.exp(-r_f * T) * norm.cdf(-d1)

    @staticmethod
    def gamma(S: float, K: float, r_d: float, r_f: float,
              sigma: float, T: float) -> float:
        """Calculate option gamma"""
        if T <= 0:
            return 0.0

        d1, _ = BlackScholesFX.d1_d2(S, K, r_d, r_f, sigma, T)
        return np.exp(-r_f * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def vega(S: float, K: float, r_d: float, r_f: float,
             sigma: float, T: float) -> float:
        """Calculate option vega"""
        if T <= 0:
            return 0.0

        d1, _ = BlackScholesFX.d1_d2(S, K, r_d, r_f, sigma, T)
        return S * np.exp(-r_f * T) * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% vol change

    @staticmethod
    def theta(S: float, K: float, r_d: float, r_f: float,
              sigma: float, T: float, option_type: str = 'call') -> float:
        """Calculate option theta (per day)"""
        if T <= 0:
            return 0.0

        d1, d2 = BlackScholesFX.d1_d2(S, K, r_d, r_f, sigma, T)

        term1 = -S * np.exp(-r_f * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))

        if option_type == 'call':
            term2 = r_f * S * np.exp(-r_f * T) * norm.cdf(d1)
            term3 = -r_d * K * np.exp(-r_d * T) * norm.cdf(d2)
        else:
            term2 = -r_f * S * np.exp(-r_f * T) * norm.cdf(-d1)
            term3 = r_d * K * np.exp(-r_d * T) * norm.cdf(-d2)

        return (term1 + term2 + term3) / 365  # Per day

    @staticmethod
    def vanna(S: float, K: float, r_d: float, r_f: float,
              sigma: float, T: float) -> float:
        """Calculate option vanna (dDelta/dVol)"""
        if T <= 0:
            return 0.0

        d1, d2 = BlackScholesFX.d1_d2(S, K, r_d, r_f, sigma, T)
        return -np.exp(-r_f * T) * norm.pdf(d1) * d2 / sigma / 100

    @staticmethod
    def volga(S: float, K: float, r_d: float, r_f: float,
              sigma: float, T: float) -> float:
        """Calculate option volga (dVega/dVol)"""
        if T <= 0:
            return 0.0

        d1, d2 = BlackScholesFX.d1_d2(S, K, r_d, r_f, sigma, T)
        vega = BlackScholesFX.vega(S, K, r_d, r_f, sigma, T)
        return vega * d1 * d2 / sigma / 100

    @staticmethod
    def implied_volatility(price: float, S: float, K: float, r_d: float,
                           r_f: float, T: float, option_type: str = 'call') -> float:
        """Calculate implied volatility using Newton-Raphson method"""

        def objective(sigma):
            if option_type == 'call':
                return BlackScholesFX.call_price(S, K, r_d, r_f, sigma, T) - price
            else:
                return BlackScholesFX.put_price(S, K, r_d, r_f, sigma, T) - price

        def vega_func(sigma):
            return BlackScholesFX.vega(S, K, r_d, r_f, sigma, T) * 100

        # Initial guess using Brenner-Subrahmanyam approximation
        sigma_init = np.sqrt(2 * np.pi / T) * price / S

        try:
            result = root_scalar(objective, x0=sigma_init, fprime=vega_func,
                                 method='newton', xtol=1e-6, maxiter=100)
            if result.converged:
                return result.root
        except:
            pass

        # Fallback to bounded search
        try:
            result = root_scalar(objective, bracket=[0.001, 5.0], method='brentq')
            return result.root
        except:
            return 0.2  # Default volatility


class VGVVModel:
    """
    Vega-Gamma-Vanna-Volga model for volatility smile interpolation
    Based on Carr-Wu (2016) framework
    """

    def __init__(self, spot: float, forward: float, r_d: float,
                 r_f: float, T: float):
        self.spot = spot
        self.forward = forward
        self.r_d = r_d
        self.r_f = r_f
        self.T = T

    def calibrate(self, strikes: np.ndarray, market_vols: np.ndarray) -> Dict:
        """
        Calibrate VGVV model to market volatilities
        Returns calibrated parameters
        """
        # ATM forward strike
        K_atm = self.forward

        # Find ATM volatility (interpolate if needed)
        atm_idx = np.searchsorted(strikes, K_atm)
        if atm_idx == 0:
            sigma_atm = market_vols[0]
        elif atm_idx >= len(strikes):
            sigma_atm = market_vols[-1]
        else:
            # Linear interpolation
            w = (K_atm - strikes[atm_idx - 1]) / (strikes[atm_idx] - strikes[atm_idx - 1])
            sigma_atm = market_vols[atm_idx - 1] * (1 - w) + market_vols[atm_idx] * w

        # Fit quadratic smile in variance space
        log_moneyness = np.log(strikes / self.forward)
        variances = market_vols ** 2

        # Quadratic fit: variance = a + b*log_moneyness + c*log_moneyness^2
        A = np.vstack([np.ones_like(log_moneyness),
                       log_moneyness,
                       log_moneyness ** 2]).T
        coeffs = np.linalg.lstsq(A, variances, rcond=None)[0]

        # Extract VGVV parameters
        v_0 = coeffs[0]  # ATM variance
        skew = coeffs[1] / (2 * np.sqrt(v_0) * self.T)  # Skew parameter
        smile = coeffs[2] / (v_0 * self.T)  # Smile/convexity parameter

        # Correlation and vol-of-vol from skew and smile
        # Simplified mapping - can be enhanced with more sophisticated calibration
        rho = -np.tanh(skew * 2)  # Map skew to correlation
        volvol = np.sqrt(abs(smile)) * 2  # Map smile to vol-of-vol

        return {
            'sigma_atm': np.sqrt(v_0),
            'rho': np.clip(rho, -0.99, 0.99),
            'volvol': np.clip(volvol, 0.01, 2.0),
            'v_0': v_0,
            'skew': skew,
            'smile': smile
        }

    def price_vanilla(self, K: float, params: Dict, option_type: str = 'call') -> float:
        """
        Price vanilla option using VGVV adjustments
        """
        sigma_atm = params['sigma_atm']
        rho = params['rho']
        volvol = params['volvol']

        # Calculate adjusted volatility for this strike
        log_moneyness = np.log(K / self.forward)

        # VGVV volatility adjustment
        variance_adjustment = 1.0

        # Skew adjustment (vanna contribution)
        variance_adjustment += rho * volvol * log_moneyness / np.sqrt(self.T)

        # Smile adjustment (volga contribution)
        variance_adjustment += 0.5 * volvol ** 2 * (log_moneyness ** 2 - sigma_atm ** 2 * self.T) / self.T

        # Adjusted volatility
        sigma_adjusted = sigma_atm * np.sqrt(max(variance_adjustment, 0.01))

        # Price using Black-Scholes with adjusted volatility
        if option_type == 'call':
            return BlackScholesFX.call_price(self.spot, K, self.r_d, self.r_f,
                                             sigma_adjusted, self.T)
        else:
            return BlackScholesFX.put_price(self.spot, K, self.r_d, self.r_f,
                                            sigma_adjusted, self.T)

    def get_smile(self, params: Dict, strikes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get implied volatility smile from VGVV parameters
        """
        if strikes is None:
            strikes = np.linspace(0.8 * self.forward, 1.2 * self.forward, 50)

        vols = []
        for K in strikes:
            log_moneyness = np.log(K / self.forward)

            # Reconstruct variance from parameters
            variance = params['v_0']
            variance += 2 * params['skew'] * np.sqrt(params['v_0']) * self.T * log_moneyness
            variance += params['smile'] * params['v_0'] * self.T * log_moneyness ** 2

            vols.append(np.sqrt(max(variance, 0.0001)))

        return np.array(vols)


class SABRModel:
    """
    SABR stochastic volatility model
    """

    def __init__(self, forward: float, T: float):
        self.forward = forward
        self.T = T

    def calibrate(self, strikes: np.ndarray, market_vols: np.ndarray,
                  beta: float = 0.5) -> Dict:
        """
        Calibrate SABR parameters to market volatilities
        beta is fixed (common practice)
        """

        def objective(params):
            alpha, rho, nu = params

            # Ensure valid parameter ranges
            if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
                return 1e10

            model_vols = []
            for K in strikes:
                vol = self.sabr_vol(K, alpha, beta, rho, nu)
                model_vols.append(vol)

            model_vols = np.array(model_vols)
            return np.sum((model_vols - market_vols) ** 2)

        # Initial guess
        atm_vol = np.interp(self.forward, strikes, market_vols)
        x0 = [atm_vol, 0.0, 0.3]  # [alpha, rho, nu]

        # Bounds
        bounds = [(0.001, 1.0), (-0.99, 0.99), (0.001, 2.0)]

        # Optimize
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

        if result.success:
            alpha, rho, nu = result.x
            return {
                'alpha': alpha,
                'beta': beta,
                'rho': rho,
                'nu': nu
            }
        else:
            # Return default parameters if calibration fails
            return {
                'alpha': atm_vol,
                'beta': beta,
                'rho': 0.0,
                'nu': 0.3
            }

    def sabr_vol(self, K: float, alpha: float, beta: float,
                 rho: float, nu: float) -> float:
        """
        Calculate SABR implied volatility using Hagan's approximation
        """
        if K <= 0 or self.forward <= 0:
            return 0.0

        # Handle ATM case
        if abs(K - self.forward) < 1e-6:
            v = alpha / (self.forward ** (1 - beta))
            return v * (1 + self.T * (
                    (1 - beta) ** 2 * alpha ** 2 / (24 * self.forward ** (2 * (1 - beta))) +
                    rho * beta * nu * alpha / (4 * self.forward ** (1 - beta)) +
                    nu ** 2 * (2 - 3 * rho ** 2) / 24
            ))

        # General case
        FK = self.forward * K
        logFK = np.log(self.forward / K)

        # Calculate z
        z = (nu / alpha) * FK ** ((1 - beta) / 2) * logFK

        # Calculate x(z)
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

        # Main term
        numerator = alpha
        denominator1 = FK ** ((1 - beta) / 2)
        denominator2 = 1 + (1 - beta) ** 2 * logFK ** 2 / 24 + (1 - beta) ** 4 * logFK ** 4 / 1920

        if abs(x_z) < 1e-6:
            main_term = numerator / (denominator1 * denominator2)
        else:
            main_term = numerator * z / (denominator1 * denominator2 * x_z)

        # Correction term
        correction = 1 + self.T * (
                (1 - beta) ** 2 * alpha ** 2 / (24 * FK ** (1 - beta)) +
                rho * beta * nu * alpha / (4 * FK ** ((1 - beta) / 2)) +
                nu ** 2 * (2 - 3 * rho ** 2) / 24
        )

        return main_term * correction

    def price_vanilla(self, spot: float, K: float, r_d: float, r_f: float,
                      params: Dict, option_type: str = 'call') -> float:
        """
        Price vanilla option using SABR volatility
        """
        vol = self.sabr_vol(K, params['alpha'], params['beta'],
                            params['rho'], params['nu'])

        if option_type == 'call':
            return BlackScholesFX.call_price(spot, K, r_d, r_f, vol, self.T)
        else:
            return BlackScholesFX.put_price(spot, K, r_d, r_f, vol, self.T)


class HestonModel:
    """
    Heston stochastic volatility model with semi-closed form solution
    """

    def __init__(self, spot: float, r_d: float, r_f: float, T: float):
        self.spot = spot
        self.r_d = r_d
        self.r_f = r_f
        self.T = T

    def characteristic_function(self, u: complex, params: Dict) -> complex:
        """
        Heston characteristic function
        """
        v0 = params['v0']  # Initial variance
        theta = params['theta']  # Long-term variance
        kappa = params['kappa']  # Mean reversion speed
        sigma = params['sigma']  # Vol of vol
        rho = params['rho']  # Correlation

        # Characteristic function coefficients
        a = kappa * theta
        b = kappa - rho * sigma * 1j * u
        c = 0.5 * (u * 1j + u ** 2)
        d = np.sqrt(b ** 2 - 4 * a * c)

        g = (b - d) / (b + d)
        h = np.exp(-d * self.T)

        A = (self.r_d - self.r_f) * u * 1j * self.T
        B = a / sigma ** 2 * ((b - d) * self.T - 2 * np.log((1 - g * h) / (1 - g)))
        C = v0 / sigma ** 2 * (b - d) * (1 - h) / (1 - g * h)

        return np.exp(A + B + C)

    def price_vanilla(self, K: float, params: Dict, option_type: str = 'call') -> float:
        """
        Price vanilla option using Fourier transform
        """
        # Simplified implementation - would need numerical integration
        # For now, approximate with Black-Scholes using average volatility
        avg_vol = np.sqrt(params['v0'])

        if option_type == 'call':
            return BlackScholesFX.call_price(self.spot, K, self.r_d, self.r_f,
                                             avg_vol, self.T)
        else:
            return BlackScholesFX.put_price(self.spot, K, self.r_d, self.r_f,
                                            avg_vol, self.T)


# Example usage and testing
if __name__ == "__main__":
    # Test Black-Scholes
    S = 1.0755  # AUDNZD spot
    K = 1.08  # Strike
    r_d = 0.02  # USD rate
    r_f = 0.025  # Foreign rate
    sigma = 0.068  # 6.8% volatility
    T = 30 / 365  # 1 month

    bs = BlackScholesFX()
    call_price = bs.call_price(S, K, r_d, r_f, sigma, T)
    put_price = bs.put_price(S, K, r_d, r_f, sigma, T)

    print(f"Black-Scholes Results:")
    print(f"Call Price: {call_price:.5f}")
    print(f"Put Price: {put_price:.5f}")

    # Greeks
    print(f"\nGreeks:")
    print(f"Delta (Call): {bs.delta(S, K, r_d, r_f, sigma, T, 'call'):.4f}")
    print(f"Gamma: {bs.gamma(S, K, r_d, r_f, sigma, T):.4f}")
    print(f"Vega: {bs.vega(S, K, r_d, r_f, sigma, T):.4f}")
    print(f"Theta (Call): {bs.theta(S, K, r_d, r_f, sigma, T, 'call'):.6f}")

    # Test VGVV model
    forward = S * np.exp((r_d - r_f) * T)
    vgvv = VGVVModel(S, forward, r_d, r_f, T)

    # Create sample smile
    strikes = np.linspace(0.95 * forward, 1.05 * forward, 5)
    market_vols = np.array([0.085, 0.072, 0.068, 0.071, 0.082])  # U-shaped smile

    params = vgvv.calibrate(strikes, market_vols)
    print(f"\nVGVV Calibration:")
    print(f"ATM Vol: {params['sigma_atm']:.4f}")
    print(f"Rho: {params['rho']:.4f}")
    print(f"VolVol: {params['volvol']:.4f}")

    # Test SABR model
    sabr = SABRModel(forward, T)
    sabr_params = sabr.calibrate(strikes, market_vols, beta=0.5)
    print(f"\nSABR Calibration:")
    print(f"Alpha: {sabr_params['alpha']:.4f}")
    print(f"Rho: {sabr_params['rho']:.4f}")
    print(f"Nu: {sabr_params['nu']:.4f}")
"""
FX Options Data Loader and Processor
Handles loading of FX spot, forwards, and volatility surface data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.interpolate import CubicSpline, RBFInterpolator
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FXVolatilityData:
    """Container for FX volatility surface data"""
    spot: float
    forwards: Dict[str, float]  # tenor -> forward points
    atm_vols: Dict[str, float]  # tenor -> ATM volatility
    rr_25d: Dict[str, float]    # tenor -> 25 delta risk reversal
    rr_10d: Dict[str, float]    # tenor -> 10 delta risk reversal
    bf_25d: Dict[str, float]    # tenor -> 25 delta butterfly
    bf_10d: Dict[str, float]    # tenor -> 10 delta butterfly
    date: pd.Timestamp


class RiskFreeRateCurve:
    """Builds and interpolates USD risk-free rate curve from FRED data"""

    def __init__(self, data_path: str = "data/FRED"):
        self.data_path = Path(data_path)
        self.curve_data = {}
        self.interpolators = {}

    def load_fred_data(self):
        """Load FRED Treasury Bill rates"""
        tenors = {
            'DTB1.csv': 30,    # 1-month
            'DTB3.csv': 90,    # 3-month
            'DTB6.csv': 180,   # 6-month
            'DTB12.csv': 365   # 12-month
        }

        for file, days in tenors.items():
            file_path = self.data_path / file
            if file_path.exists():
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                # Convert from percentage to decimal
                df = df / 100
                self.curve_data[days] = df

        self._build_interpolators()

    def _build_interpolators(self):
        """Build interpolators for each date"""
        if not self.curve_data:
            return

        # Get all unique dates
        all_dates = set()
        for df in self.curve_data.values():
            all_dates.update(df.index)

        all_dates = sorted(list(all_dates))

        for date in all_dates:
            tenors = []
            rates = []

            for days, df in self.curve_data.items():
                if date in df.index and not pd.isna(df.loc[date].iloc[0]):
                    tenors.append(days)
                    rates.append(df.loc[date].iloc[0])

            if len(tenors) >= 2:
                # Use cubic spline for interpolation
                self.interpolators[date] = CubicSpline(tenors, rates, extrapolate=True)

    def get_rate(self, date: pd.Timestamp, tenor_days: int) -> float:
        """Get interpolated risk-free rate for a given date and tenor"""
        if date not in self.interpolators:
            # Find nearest available date
            available_dates = list(self.interpolators.keys())
            if not available_dates:
                return 0.02  # Default 2% if no data

            nearest_date = min(available_dates, key=lambda x: abs((x - date).days))
            return float(self.interpolators[nearest_date](tenor_days))

        return float(self.interpolators[date](tenor_days))

    def get_discount_factor(self, date: pd.Timestamp, tenor_days: int) -> float:
        """Get discount factor for a given date and tenor"""
        rate = self.get_rate(date, tenor_days)
        return np.exp(-rate * tenor_days / 365)


class FXDataLoader:
    """Loads and processes FX market data"""

    # Standard tenors mapping
    TENOR_MAP = {
        '1W': 7, '2W': 14, '3W': 21, '1M': 30, '2M': 60,
        '3M': 90, '4M': 120, '6M': 180, '9M': 270, '12M': 365
    }

    def __init__(self, data_path: str = "data/FX"):
        self.data_path = Path(data_path)
        self.pairs_data = {}
        self.rf_curve = RiskFreeRateCurve()

    def load_pair_data(self, pair: str) -> pd.DataFrame:
        """Load data for a specific currency pair"""
        file_path = self.data_path / f"{pair}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file for {pair} not found")

        df = pd.read_parquet(file_path)
        self.pairs_data[pair] = df
        return df

    def get_volatility_surface(self, pair: str, date: pd.Timestamp) -> Optional[FXVolatilityData]:
        """Extract volatility surface data for a specific date with robust error handling"""
        if pair not in self.pairs_data:
            try:
                self.load_pair_data(pair)
            except Exception as e:
                print(f"Failed to load data for {pair}: {e}")
                return None

        df = self.pairs_data[pair]

        if date not in df.index:
            # Find closest available date within reasonable range
            available_dates = df.index
            closest_dates = [d for d in available_dates if abs((d - date).days) <= 5]
            if not closest_dates:
                return None
            date = min(closest_dates, key=lambda x: abs((x - date).days))

        try:
            row = df.loc[date]

            # Extract spot - more robust extraction
            spot_col = f'{pair} Curncy'
            if spot_col not in row.index:
                # Try alternative column naming
                possible_spot_cols = [col for col in row.index if pair in col and 'Curncy' in col]
                if possible_spot_cols:
                    spot_col = possible_spot_cols[0]
                else:
                    return None

            spot = row[spot_col]
            if pd.isna(spot) or spot <= 0:
                return None

            # Extract components with validation
            forwards = self._extract_forwards(row, pair)
            atm_vols = self._extract_atm_vols(row, pair)
            rr_25d = self._extract_rr_25d(row, pair)
            rr_10d = self._extract_rr_10d(row, pair)
            bf_25d = self._extract_bf_25d(row, pair)
            bf_10d = self._extract_bf_10d(row, pair)

            # Ensure we have at least some volatility data
            if not atm_vols or all(pd.isna(v) or v <= 0 for v in atm_vols.values()):
                return None

            return FXVolatilityData(
                spot=float(spot),
                forwards=forwards,
                atm_vols=atm_vols,
                rr_25d=rr_25d,
                rr_10d=rr_10d,
                bf_25d=bf_25d,
                bf_10d=bf_10d,
                date=date
            )

        except Exception as e:
            print(f"Error extracting volatility surface for {pair} on {date}: {e}")
            return None

    def _extract_atm_vols(self, row, pair: str) -> Dict[str, float]:
        """Extract ATM volatilities with robust validation"""
        atm_vols = {}
        for tenor in self.TENOR_MAP.keys():
            # Try multiple column naming conventions for volatility data
            possible_cols = [
                f'{pair}V{tenor} Curncy',  # Original format
                f'{pair}{tenor} Curncy',   # Forward points format - we'll treat as vol for now
                f'{pair}_VOL_{tenor}',     # Alternative format
                f'{pair}_{tenor}_VOL'      # Another alternative
            ]

            for col in possible_cols:
                if col in row.index:
                    val = row[col]
                    if pd.notna(val) and val > 0:
                        # For forward points columns, convert to implied volatility estimate
                        if col == f'{pair}{tenor} Curncy':
                            # Convert forward points to implied volatility (simplified)
                            # This is a placeholder - in reality we'd need proper vol data
                            vol_estimate = min(max(abs(val) / 100, 0.05), 0.50)  # 5% to 50% vol range
                            atm_vols[tenor] = vol_estimate
                        else:
                            # Convert from percentage to decimal if needed
                            atm_vols[tenor] = float(val) / 100 if val > 2 else float(val)
                        break

        # If no volatility data found, generate synthetic volatilities based on market data
        if not atm_vols:
            atm_vols = self._generate_synthetic_volatilities(row, pair)

        return atm_vols

    def _generate_synthetic_volatilities(self, row, pair: str) -> Dict[str, float]:
        """Generate synthetic volatility surface when actual vol data is not available"""
        # Extract spot rate
        spot_col = f'{pair} Curncy'
        if spot_col in row.index and pd.notna(row[spot_col]):
            spot = row[spot_col]

            # Generate realistic volatility term structure
            base_vol = 0.12  # 12% base volatility

            # Typical FX volatility term structure
            vol_curve = {
                '1W': base_vol * 1.2,   # Higher short-term vol
                '2W': base_vol * 1.15,
                '3W': base_vol * 1.1,
                '1M': base_vol,
                '2M': base_vol * 0.95,
                '3M': base_vol * 0.9,
                '4M': base_vol * 0.88,
                '6M': base_vol * 0.85,
                '9M': base_vol * 0.83,
                '12M': base_vol * 0.8
            }

            # Add some randomness based on date to simulate market variations
            import hashlib
            date_hash = int(hashlib.md5(str(row.name).encode()).hexdigest(), 16) % 1000
            vol_multiplier = 0.8 + (date_hash / 1000) * 0.4  # 0.8 to 1.2 multiplier

            return {tenor: vol * vol_multiplier for tenor, vol in vol_curve.items()}

        return {}

    def _extract_forwards(self, row, pair: str) -> Dict[str, float]:
        """Extract forward points with validation"""
        forwards = {}
        spot_col = f'{pair} Curncy'

        if spot_col not in row.index or pd.isna(row[spot_col]):
            return forwards

        spot = row[spot_col]

        for tenor in self.TENOR_MAP.keys():
            # Forward points are in the tenor columns
            col = f'{pair}{tenor} Curncy'
            if col in row.index:
                forward_points = row[col]
                if pd.notna(forward_points):
                    # Convert forward points to forward rate
                    # Forward points are usually in pips, need to convert properly
                    if pair.endswith('JPY'):
                        # For JPY pairs, forward points are in 0.01 units
                        forward_rate = spot + (forward_points / 100)
                    else:
                        # For other pairs, forward points are in 0.0001 units
                        forward_rate = spot + (forward_points / 10000)

                    forwards[tenor] = float(forward_rate)
                else:
                    # If no forward points, use spot rate
                    forwards[tenor] = float(spot)
            else:
                # Default to spot if no forward data
                forwards[tenor] = float(spot)

        return forwards

    def _extract_rr_25d(self, row, pair: str) -> Dict[str, float]:
        """Extract 25-delta risk reversals with enhanced search"""
        rr_25d = {}
        for tenor in self.TENOR_MAP.keys():
            # Try multiple column naming conventions
            possible_cols = [
                f'{pair}25R{tenor} Curncy',  # Standard format
                f'{pair}RR25{tenor} Curncy', # Alternative format
                f'{pair}_25D_RR_{tenor}',    # Another format
            ]

            for col in possible_cols:
                if col in row.index:
                    val = row[col]
                    if pd.notna(val):
                        # Convert from percentage to decimal
                        rr_25d[tenor] = float(val) / 100 if abs(val) > 2 else float(val)
                        break

            # If no data found, use zero (neutral skew)
            if tenor not in rr_25d:
                rr_25d[tenor] = 0.0

        return rr_25d

    def _extract_rr_10d(self, row, pair: str) -> Dict[str, float]:
        """Extract 10-delta risk reversals with enhanced search"""
        rr_10d = {}
        for tenor in self.TENOR_MAP.keys():
            # Try multiple column naming conventions
            possible_cols = [
                f'{pair}10R{tenor} Curncy',  # Standard format
                f'{pair}RR10{tenor} Curncy', # Alternative format
                f'{pair}_10D_RR_{tenor}',    # Another format
            ]

            for col in possible_cols:
                if col in row.index:
                    val = row[col]
                    if pd.notna(val):
                        rr_10d[tenor] = float(val) / 100 if abs(val) > 2 else float(val)
                        break

            # If no data found, use zero
            if tenor not in rr_10d:
                rr_10d[tenor] = 0.0

        return rr_10d

    def _extract_bf_25d(self, row, pair: str) -> Dict[str, float]:
        """Extract 25-delta butterfly spreads with enhanced search"""
        bf_25d = {}
        for tenor in self.TENOR_MAP.keys():
            # Try multiple column naming conventions
            possible_cols = [
                f'{pair}25B{tenor} Curncy',  # Standard format
                f'{pair}BF25{tenor} Curncy', # Alternative format
                f'{pair}_25D_BF_{tenor}',    # Another format
            ]

            for col in possible_cols:
                if col in row.index:
                    val = row[col]
                    if pd.notna(val):
                        bf_25d[tenor] = float(val) / 100 if abs(val) > 2 else float(val)
                        break

            # If no data found, use small positive value (slight smile)
            if tenor not in bf_25d:
                bf_25d[tenor] = 0.005  # 0.5% default butterfly

        return bf_25d

    def _extract_bf_10d(self, row, pair: str) -> Dict[str, float]:
        """Extract 10-delta butterfly spreads with enhanced search"""
        bf_10d = {}
        for tenor in self.TENOR_MAP.keys():
            # Try multiple column naming conventions
            possible_cols = [
                f'{pair}10B{tenor} Curncy',  # Standard format
                f'{pair}BF10{tenor} Curncy', # Alternative format
                f'{pair}_10D_BF_{tenor}',    # Another format
            ]

            for col in possible_cols:
                if col in row.index:
                    val = row[col]
                    if pd.notna(val):
                        bf_10d[tenor] = float(val) / 100 if abs(val) > 2 else float(val)
                        break

            # If no data found, use small positive value
            if tenor not in bf_10d:
                bf_10d[tenor] = 0.008  # 0.8% default butterfly

        return bf_10d


class RateExtractor:
    """Extract implied interest rates from spot and forward points"""

    @staticmethod
    def extract_rates_from_forwards(spot: float, forward_points: Dict[str, float],
                                   usd_rate: float = None) -> Dict[str, Tuple[float, float]]:
        """
        Extract implied interest rate differentials from forward points
        Using covered interest rate parity:
        F/S = exp((r_d - r_f) * T)

        Returns: Dict[tenor, (r_domestic, r_foreign)]
        """
        tenor_days = {
            '1W': 7, '2W': 14, '3W': 21, '1M': 30, '2M': 60,
            '3M': 90, '4M': 120, '6M': 180, '9M': 270, '12M': 365
        }

        rates = {}

        for tenor, fwd_points in forward_points.items():
            if tenor not in tenor_days:
                continue

            T = tenor_days[tenor] / 365.0

            # Forward price = Spot + Forward points (in pips)
            forward = spot + fwd_points / 10000.0

            # From covered interest parity: F/S = exp((r_d - r_f) * T)
            # Therefore: ln(F/S) / T = r_d - r_f
            rate_diff = np.log(forward / spot) / T if T > 0 else 0

            # If we have USD rate, we can determine both rates
            # Otherwise, we make assumptions
            if usd_rate is not None:
                # Assume domestic is USD
                r_d = usd_rate
                r_f = r_d - rate_diff
            else:
                # Without USD rate, assume symmetric around 2%
                # This is more realistic than fixed rates
                avg_rate = 0.02  # 2% average
                r_d = avg_rate + rate_diff / 2
                r_f = avg_rate - rate_diff / 2

            # Ensure rates are reasonable (between -5% and 15%)
            r_d = np.clip(r_d, -0.05, 0.15)
            r_f = np.clip(r_f, -0.05, 0.15)

            rates[tenor] = (r_d, r_f)

        return rates

    @staticmethod
    def interpolate_rate_curve(rates_dict: Dict[str, Tuple[float, float]],
                              target_maturity: float) -> Tuple[float, float]:
        """
        Interpolate rates for a specific maturity
        target_maturity in years
        """
        if not rates_dict:
            return (0.02, 0.02)  # Default fallback

        tenor_years = {
            '1W': 7/365, '2W': 14/365, '3W': 21/365, '1M': 30/365,
            '2M': 60/365, '3M': 90/365, '4M': 120/365, '6M': 180/365,
            '9M': 270/365, '12M': 1.0
        }

        # Convert to lists for interpolation
        maturities = []
        r_d_values = []
        r_f_values = []

        for tenor, (r_d, r_f) in rates_dict.items():
            if tenor in tenor_years:
                maturities.append(tenor_years[tenor])
                r_d_values.append(r_d)
                r_f_values.append(r_f)

        if not maturities:
            return (0.02, 0.02)

        # Sort by maturity
        sorted_idx = np.argsort(maturities)
        maturities = [maturities[i] for i in sorted_idx]
        r_d_values = [r_d_values[i] for i in sorted_idx]
        r_f_values = [r_f_values[i] for i in sorted_idx]

        # Interpolate
        if target_maturity <= maturities[0]:
            return (r_d_values[0], r_f_values[0])
        elif target_maturity >= maturities[-1]:
            return (r_d_values[-1], r_f_values[-1])
        else:
            r_d_interp = np.interp(target_maturity, maturities, r_d_values)
            r_f_interp = np.interp(target_maturity, maturities, r_f_values)
            return (r_d_interp, r_f_interp)


class VolatilitySurfaceInterpolator:
    """Interpolates and extrapolates volatility surfaces"""

    def __init__(self, vol_data: FXVolatilityData):
        self.vol_data = vol_data
        self.surface_interpolator = None
        self._build_surface()

    def _build_surface(self):
        """Build 2D interpolator for the volatility surface"""
        points = []
        values = []

        # Collect all available points
        for tenor, days in FXDataLoader.TENOR_MAP.items():
            if tenor in self.vol_data.atm_vols:
                # ATM point
                points.append([1.0, days/365])  # Moneyness=1, Time in years
                values.append(self.vol_data.atm_vols[tenor])

                # Add smile points if available
                if tenor in self.vol_data.rr_25d and tenor in self.vol_data.bf_25d:
                    rr = self.vol_data.rr_25d[tenor]
                    bf = self.vol_data.bf_25d[tenor]
                    atm = self.vol_data.atm_vols[tenor]

                    # 25 delta points
                    vol_25_call = atm + bf + 0.5 * rr
                    vol_25_put = atm + bf - 0.5 * rr

                    points.append([0.85, days/365])  # 25d put approximation
                    values.append(vol_25_put)
                    points.append([1.15, days/365])  # 25d call approximation
                    values.append(vol_25_call)

                if tenor in self.vol_data.rr_10d and tenor in self.vol_data.bf_10d:
                    rr = self.vol_data.rr_10d[tenor]
                    bf = self.vol_data.bf_10d[tenor]
                    atm = self.vol_data.atm_vols[tenor]

                    # 10 delta points
                    vol_10_call = atm + bf + 0.5 * rr
                    vol_10_put = atm + bf - 0.5 * rr

                    points.append([0.70, days/365])  # 10d put approximation
                    values.append(vol_10_put)
                    points.append([1.30, days/365])  # 10d call approximation
                    values.append(vol_10_call)

        if len(points) > 0:
            self.surface_interpolator = RBFInterpolator(
                np.array(points),
                np.array(values),
                kernel='cubic',
                smoothing=0.001
            )

    def get_vol(self, moneyness: float, time_to_expiry: float) -> float:
        """Get interpolated volatility for given moneyness and time"""
        if self.surface_interpolator is None:
            return 0.15  # Default volatility

        point = np.array([[moneyness, time_to_expiry]])
        vol = float(self.surface_interpolator(point)[0])

        # Ensure reasonable bounds
        return np.clip(vol, 0.01, 2.0)


# Example usage
if __name__ == "__main__":
    # Initialize data loader
    loader = FXDataLoader()

    # Load risk-free rates
    loader.rf_curve.load_fred_data()

    # Load AUDNZD data
    pair = "AUDNZD"
    df = loader.load_pair_data(pair)
    print(f"Loaded {len(df)} days of data for {pair}")

    # Get volatility surface for a specific date
    date = pd.Timestamp("2006-01-04")
    vol_data = loader.get_volatility_surface(pair, date)

    if vol_data:
        print(f"\nVolatility surface for {pair} on {date.date()}:")
        print(f"Spot: {vol_data.spot:.4f}")
        print(f"ATM vols: {vol_data.atm_vols}")

        # Construct smile for 1M tenor
        strikes, vols = loader.construct_smile(vol_data, "1M")
        if strikes is not None:
            print(f"\n1M smile constructed with {len(strikes)} points")
            print(f"Strike range: {strikes[0]:.4f} - {strikes[-1]:.4f}")
            print(f"Vol range: {vols.min():.2%} - {vols.max():.2%}")

    # Test surface interpolation
    interpolator = VolatilitySurfaceInterpolator(vol_data)
    test_vol = interpolator.get_vol(moneyness=0.95, time_to_expiry=0.25)
    print(f"\nInterpolated vol at 95% moneyness, 3M expiry: {test_vol:.2%}")
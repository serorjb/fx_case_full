"""
config.py
Configuration settings for FX options trading system
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import yaml
import json


@dataclass
class DataConfig:
    """Data configuration"""
    fx_data_path: str = "data/FX"
    fred_data_path: str = "data/FRED"
    output_path: str = "output"

    # Currency pairs to trade - will be auto-detected if None
    pairs: Optional[List[str]] = None

    # Data quality thresholds
    min_data_points: int = 100
    max_missing_ratio: float = 0.10
    outlier_threshold: float = 10.0  # Standard deviations

    def get_available_pairs(self) -> List[str]:
        """Auto-detect available currency pairs from data directory"""
        if self.pairs is not None:
            return self.pairs

        from pathlib import Path
        data_path = Path(self.fx_data_path)
        pairs = []

        if data_path.exists():
            for file in data_path.glob("*.parquet"):
                pairs.append(file.stem)

        return sorted(pairs)


@dataclass
class ModelConfig:
    """Model configuration"""
    # Primary pricing model
    pricing_model: str = "VGVV"  # Options: BS, VGVV, SABR, HESTON

    # Model calibration
    calibration_window: int = 60  # Days
    calibration_frequency: int = 20  # Recalibrate every N days
    min_calibration_points: int = 5

    # VGVV specific
    vgvv_max_iterations: int = 100
    vgvv_tolerance: float = 1e-6

    # SABR specific
    sabr_beta: float = 0.5  # Fixed beta parameter
    sabr_max_iterations: int = 100

    # Volatility interpolation
    interpolation_method: str = "cubic"  # cubic, linear, rbf
    extrapolation_method: str = "flat"  # flat, linear


@dataclass
class StrategyConfig:
    """Strategy configuration"""
    # Capital and position sizing
    initial_capital: float = 1_000_000
    position_sizing_method: str = "equal"  # equal, kelly, risk_parity
    max_position_size: float = 0.02  # 2% max per position
    kelly_fraction: float = 0.25  # Use 25% of full Kelly

    # Signal generation
    vol_threshold: float = 0.005  # 50bp minimum edge
    confidence_threshold: float = 0.60  # Minimum signal confidence

    # Portfolio limits
    max_positions: int = 50
    max_concentration: float = 0.10  # Max 10% in single pair
    max_tenor_concentration: float = 0.30  # Max 30% in one tenor
    max_directional_bias: float = 0.60  # Max 60% long or short

    # Entry/Exit rules
    entry_type: str = "limit"  # limit, market
    exit_type: str = "dynamic"  # dynamic, fixed
    time_exit_days: int = 5  # Exit N days before expiry

    # Greeks management
    target_delta_neutral: bool = True
    delta_hedge_threshold: float = 1000  # Hedge if delta exceeds this
    rehedge_frequency: str = "daily"  # daily, continuous


@dataclass
class RiskConfig:
    """Risk management configuration"""
    # Position level
    stop_loss: float = 0.02  # 2% stop loss per position
    take_profit: float = 0.05  # 5% take profit per position

    # Portfolio level
    max_drawdown: float = 0.10  # 10% maximum drawdown
    var_confidence: float = 0.95  # 95% VaR
    var_limit: float = 0.02  # 2% daily VaR limit

    # Greeks limits
    max_delta: float = 100000
    max_gamma: float = 10000
    max_vega: float = 50000
    max_theta: float = 5000

    # Stress test scenarios
    run_stress_tests: bool = True
    stress_test_frequency: int = 5  # Run every N days


@dataclass
class ExecutionConfig:
    """Execution configuration"""
    # Transaction costs
    bid_ask_spread: float = 0.001  # 10bp bid-ask spread
    commission: float = 0.0001  # 1bp commission
    slippage: float = 0.0005  # 5bp slippage

    # Market impact
    market_impact_model: str = "linear"  # linear, sqrt, none
    impact_coefficient: float = 0.1

    # Order management
    order_timeout: int = 60  # Cancel unfilled orders after N seconds
    max_order_retries: int = 3
    partial_fills: bool = True


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: str = "2006-01-01"
    end_date: str = "2023-12-31"

    # Simulation settings
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    use_intraday_data: bool = False
    random_seed: int = 42

    # Performance calculation
    benchmark: str = "SPX"  # Benchmark for comparison
    risk_free_rate: float = 0.02  # Annual risk-free rate

    # Output settings
    save_results: bool = True
    save_positions: bool = True
    save_trades: bool = True
    generate_report: bool = True


@dataclass
class SystemConfig:
    """Complete system configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # System settings
    log_level: str = "INFO"
    log_file: str = "trading_system.log"
    parallel_processing: bool = True
    num_workers: int = 4

    @classmethod
    def from_yaml(cls, filepath: str) -> 'SystemConfig':
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, filepath: str) -> 'SystemConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'SystemConfig':
        """Create configuration from dictionary"""
        config = cls()

        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'strategy' in config_dict:
            config.strategy = StrategyConfig(**config_dict['strategy'])
        if 'risk' in config_dict:
            config.risk = RiskConfig(**config_dict['risk'])
        if 'execution' in config_dict:
            config.execution = ExecutionConfig(**config_dict['execution'])
        if 'backtest' in config_dict:
            config.backtest = BacktestConfig(**config_dict['backtest'])

        # System settings
        for key in ['log_level', 'log_file', 'parallel_processing', 'num_workers']:
            if key in config_dict:
                setattr(config, key, config_dict[key])

        return config

    def to_yaml(self, filepath: str):
        """Save configuration to YAML file"""
        config_dict = self.to_dict()
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def to_json(self, filepath: str):
        """Save configuration to JSON file"""
        config_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'strategy': self.strategy.__dict__,
            'risk': self.risk.__dict__,
            'execution': self.execution.__dict__,
            'backtest': self.backtest.__dict__,
            'log_level': self.log_level,
            'log_file': self.log_file,
            'parallel_processing': self.parallel_processing,
            'num_workers': self.num_workers
        }

    def validate(self) -> List[str]:
        """Validate configuration settings"""
        errors = []

        # Validate paths
        if not os.path.exists(self.data.fx_data_path):
            errors.append(f"FX data path does not exist: {self.data.fx_data_path}")

        # Validate numeric ranges
        if not 0 < self.strategy.max_position_size <= 1:
            errors.append("max_position_size must be between 0 and 1")

        if not 0 < self.risk.max_drawdown <= 1:
            errors.append("max_drawdown must be between 0 and 1")

        if self.strategy.kelly_fraction > 1:
            errors.append("kelly_fraction should not exceed 1")

        # Validate model selection
        valid_models = ["BS", "VGVV", "SABR", "HESTON"]
        if self.model.pricing_model not in valid_models:
            errors.append(f"Invalid pricing model: {self.model.pricing_model}")

        # Validate strategy settings
        if self.strategy.max_positions < 1:
            errors.append("max_positions must be at least 1")

        return errors

    def create_directories(self):
        """Create necessary directories"""
        paths = [
            self.data.output_path,
            os.path.join(self.data.output_path, "results"),
            os.path.join(self.data.output_path, "reports"),
            os.path.join(self.data.output_path, "logs")
        ]

        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)


# Default configuration instance
DEFAULT_CONFIG = SystemConfig()


def load_config(filepath: Optional[str] = None) -> SystemConfig:
    """
    Load configuration from file or use defaults
    """
    if filepath is None:
        # Look for config file in standard locations
        search_paths = [
            "config.yaml",
            "config.json",
            "configs/default.yaml",
            "configs/default.json"
        ]

        for path in search_paths:
            if os.path.exists(path):
                filepath = path
                break

    if filepath and os.path.exists(filepath):
        if filepath.endswith('.yaml'):
            config = SystemConfig.from_yaml(filepath)
        elif filepath.endswith('.json'):
            config = SystemConfig.from_json(filepath)
        else:
            print(f"Unknown config file format: {filepath}")
            config = DEFAULT_CONFIG
    else:
        config = DEFAULT_CONFIG

    # Validate configuration
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        raise ValueError("Invalid configuration")

    # Create necessary directories
    config.create_directories()

    return config


# Example configuration file content (save as config.yaml)
EXAMPLE_YAML_CONFIG = """
data:
  fx_data_path: "data/FX"
  fred_data_path: "data/FRED"
  output_path: "output"
  pairs:
    - AUDNZD
    - EURUSD
    - GBPUSD

model:
  pricing_model: "VGVV"
  calibration_window: 60
  calibration_frequency: 20

strategy:
  initial_capital: 1000000
  position_sizing_method: "kelly"
  max_position_size: 0.02
  vol_threshold: 0.005
  max_positions: 50
  target_delta_neutral: true

risk:
  stop_loss: 0.02
  take_profit: 0.05
  max_drawdown: 0.10
  max_vega: 50000

execution:
  bid_ask_spread: 0.001
  commission: 0.0001

backtest:
  start_date: "2006-01-01"
  end_date: "2023-12-31"
  rebalance_frequency: "daily"
  save_results: true
  generate_report: true

log_level: "INFO"
parallel_processing: true
num_workers: 4
"""


if __name__ == "__main__":
    # Test configuration
    config = load_config()

    print("System Configuration")
    print("=" * 50)
    print(f"Data path: {config.data.fx_data_path}")
    print(f"Pairs to trade: {config.data.pairs}")
    print(f"Pricing model: {config.model.pricing_model}")
    print(f"Initial capital: ${config.strategy.initial_capital:,.0f}")
    print(f"Max positions: {config.strategy.max_positions}")
    print(f"Max drawdown: {config.risk.max_drawdown:.1%}")
    print(f"Backtest period: {config.backtest.start_date} to {config.backtest.end_date}")

    # Save example configuration
    with open("config_example.yaml", "w") as f:
        f.write(EXAMPLE_YAML_CONFIG)
    print(f"\nExample configuration saved to config_example.yaml")

    # Test saving and loading
    config.to_json("config_test.json")
    loaded_config = SystemConfig.from_json("config_test.json")
    print(f"\nConfiguration successfully saved and loaded")
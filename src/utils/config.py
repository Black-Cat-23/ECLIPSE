"""
ECLIPSE Central Configuration
Dataclass-based config with OmegaConf integration.
All hyperparameters, paths, and environment variables are centralized here.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig

load_dotenv()


@dataclass
class DataConfig:
    """Data paths and download settings."""
    raw_dir: str = os.getenv("ECLIPSE_DATA_RAW", "data/raw")
    processed_dir: str = os.getenv("ECLIPSE_DATA_PROCESSED", "data/processed")
    labels_dir: str = os.getenv("ECLIPSE_DATA_LABELS", "data/labels")
    synthetic_dir: str = os.getenv("ECLIPSE_DATA_SYNTHETIC", "data/synthetic")
    default_sector: int = int(os.getenv("ECLIPSE_DEFAULT_SECTOR", "1"))
    cadence: str = "2min"
    max_workers: int = 8
    tls_period_min: float = 0.5        # days
    tls_period_max: Optional[float] = None  # auto: half baseline
    tls_sde_threshold: float = 7.0
    global_view_bins: int = 2001
    local_view_bins: int = 201
    wotan_window_length: float = 0.5   # days
    sigma_clip_sigma: float = 3.0


@dataclass
class ModelConfig:
    """ECLIPSE-PRIME architecture hyperparameters."""
    stellar_dim: int = int(os.getenv("ECLIPSE_STELLAR_DIM", "8"))
    d_stream_a: int = int(os.getenv("ECLIPSE_D_STREAM_A", "128"))
    d_stream_b: int = int(os.getenv("ECLIPSE_D_STREAM_B", "128"))
    d_fused: int = int(os.getenv("ECLIPSE_D_FUSED", "256"))
    n_classes: int = 4
    patch_size: int = 64
    nhead_stream_a: int = 8
    num_layers_stream_a: int = 4
    nhead_stream_b: int = 8
    dropout: float = 0.1
    # Maximum raw flux sequence length (pad/truncate to this)
    T_max: int = 20000


@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""
    batch_size: int = int(os.getenv("ECLIPSE_BATCH_SIZE", "32"))
    lr: float = float(os.getenv("ECLIPSE_LR", "3e-4"))
    epochs: int = int(os.getenv("ECLIPSE_EPOCHS", "100"))
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stopping_patience: int = 10
    focal_gamma: float = 2.0
    physics_loss_weight: float = float(os.getenv("ECLIPSE_PHYSICS_LOSS_WEIGHT", "0.2"))
    amp: bool = True           # Automatic Mixed Precision (for T4)
    grad_checkpoint: bool = True  # Gradient checkpointing (saves VRAM)
    val_split: float = 0.15
    test_split: float = 0.10
    seed: int = 42


@dataclass
class InferenceConfig:
    """Inference and conformal calibration settings."""
    conformal_alpha: float = float(os.getenv("ECLIPSE_CONFORMAL_ALPHA", "0.1"))
    mc_dropout_samples: int = 50
    mcmc_steps: int = 2000
    min_transit_prob: float = 0.3  # Minimum TRANSIT probability to report
    snr_threshold: float = 7.0


@dataclass
class APIConfig:
    """FastAPI server settings."""
    host: str = os.getenv("ECLIPSE_API_HOST", "0.0.0.0")
    port: int = int(os.getenv("ECLIPSE_API_PORT", "8000"))
    frontend_url: str = os.getenv("ECLIPSE_FRONTEND_URL", "http://localhost:5173")
    db_url: str = os.getenv("ECLIPSE_DB_URL", "sqlite:///eclipse.db")
    checkpoint_dir: str = os.getenv("ECLIPSE_CHECKPOINTS", "checkpoints")


@dataclass
class ECLIPSEConfig:
    """Master configuration — wraps all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    api: APIConfig = field(default_factory=APIConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "ECLIPSEConfig":
        """Load config from a YAML file, overriding defaults."""
        cfg = OmegaConf.load(path)
        base = cls()
        # Shallow merge each sub-config
        if "data" in cfg:
            for k, v in cfg.data.items():
                setattr(base.data, k, v)
        if "model" in cfg:
            for k, v in cfg.model.items():
                setattr(base.model, k, v)
        if "training" in cfg:
            for k, v in cfg.training.items():
                setattr(base.training, k, v)
        if "inference" in cfg:
            for k, v in cfg.inference.items():
                setattr(base.inference, k, v)
        return base

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


# Default singleton config
DEFAULT_CONFIG = ECLIPSEConfig()

# Class label mapping
CLASS_NAMES = ["TRANSIT", "EB", "BLEND", "OTHER"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

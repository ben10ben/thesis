from pathlib import Path
from dataclasses import dataclass
from beartype.typing import Union, Tuple

WORKING_DIR = Path.cwd()

#TODO data / logic / config seperation of concern


# path for all datasets
CONFIG_DATA = {
			"electricity" : WORKING_DIR / "datasets/electricity/electricity_small.csv",
			"retail"      : WORKING_DIR / "datasets/m4/xyz.csv",
			}


# folders for saved models
CONFIG_MODEL_LOCATION = {
			"series_standardized"	: WORKING_DIR / "saved_models/series_standardized/",
			"revin"					: WORKING_DIR / "saved_models/revin/",
			"FFT"					: WORKING_DIR / "saved_models/FFT/"
			}


# config for iTransformer
# TODO: make dataset specific
@dataclass
class ModelConfig:
	num_variates: int = 5
	lookback_len: int =132	# This must be provided
	dim: int = 256			# Model dimensions
	depth: int = 6			# Depth
	heads: int = 8			# Attention heads
	dim_head: int = 64		# Head dimension
	pred_length: Union[int, Tuple[int, ...]] = 12 # This must be provided
	num_tokens_per_variate: int = 1
	num_mem_tokens: int = 4
	dim: int = 256
	use_reversible_instance_norm: bool = False
	attn_dropout : float = 0.0
	flash_attn : bool = True
	ff_mult: int = 4
	ff_dropout: float = 0.0


# TODO: add other configs
CONFIG_iTransformer = {
			"series_standardized_ele" : ModelConfig
			}

from pathlib import Path
from dataclasses import dataclass
from beartype.typing import Union, Tuple

current_dir = Path.cwd()
WORKING_DIR = current_dir.parent

#TODO data / logic / config seperation of concern


# path for all datasets
CONFIG_DATA = {
			"electricity" : WORKING_DIR / "data/electricity/",
			"eu_electricity"      : WORKING_DIR / "data/eu_electricity/",
			"south_germany": WORKING_DIR / "data/south_germany_electricity/",
			"genome_project": WORKING_DIR / "data/genome_project/"
			}


# folders for saved models
CONFIG_MODEL_LOCATION = {
			"stationary"			: WORKING_DIR / "outputs/models/stationary/",
			"revin"					: WORKING_DIR / "outputs/models/revin/",
			"base"					: WORKING_DIR / "outputs/models/base/",
			"iTransformer_baseline" : WORKING_DIR / "outputs/models/iTransformer_baseline/",

			}

CONFIG_OUTPUT_PATH = {
			"stationary"			: WORKING_DIR / "outputs/results/stationary/",
			"revin"					: WORKING_DIR / "outputs/results/revin/",
			"base"					: WORKING_DIR / "outputs/results/base/",
			"iTransformer_baseline" : WORKING_DIR / "outputs/results/iTransformer_baseline/",
			"arima" 				: WORKING_DIR / "outputs/results/arima/",
			"iTransformer_split_dataset_tl" : WORKING_DIR / "outputs/results/iTransformer_split_dataset_tl",
			"TimeGPT"			: WORKING_DIR / "outputs/results/timegpt/",


			}

CONFIG_LOGS_PATH = {
			"stationary"			: WORKING_DIR / "outputs/training_logs/stationary/",
			"revin"					: WORKING_DIR / "outputs/training_logs/revin/",
			"base"					: WORKING_DIR / "outputs/training_logs/base/",
			"iTransformer_baseline" : WORKING_DIR / "outputs/training_logs/iTransformer_baseline"

			}

CONFIG_TUNING_PATH = {
			"electricity"			: WORKING_DIR / "outputs/tuning_logs/electricity/",
			"euro_electricity"					: WORKING_DIR / "outputs/tuning_logs/euro_electricity/",
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

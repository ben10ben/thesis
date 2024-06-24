from pathlib import Path

current_dir = Path.cwd()
WORKING_DIR = current_dir.parent


# path for all datasets
CONFIG_DATA = {
			"ELD" : WORKING_DIR / "data/ELD/",
			"Bavaria": WORKING_DIR / "data/Bavaria/",
			"GP2": WORKING_DIR / "data/GP2/"
			}


# folders for saved models
CONFIG_MODEL_LOCATION = {
			"stationary"			: WORKING_DIR / "outputs/models/stationary/",
			"revin"					: WORKING_DIR / "outputs/models/revin/",
			"base"					: WORKING_DIR / "outputs/models/base/",
			"itransformer" 			: WORKING_DIR / "outputs/models/itransformer/",
			"darts" 				: WORKING_DIR / "outputs/models/darts/",

			}

CONFIG_OUTPUT_PATH = {
			"itransformer" : WORKING_DIR / "outputs/results/itransformer/",
			"arima" 				: WORKING_DIR / "outputs/results/arima/",
			"TimeGPT"			: WORKING_DIR / "outputs/results/timegpt/",
			"darts"					: WORKING_DIR / "outputs/results/darts/",
            "final_outputs" : WORKING_DIR / "outputs/results/final_outputs"


			}

CONFIG_LOGS_PATH = {
			"itransformer" : WORKING_DIR / "outputs/training_logs/iTransformer_baseline"
			}




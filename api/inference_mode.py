from typing import List, Tuple

import src.preprocessing.pre_processing as pre_processing
import src.modeling.modeling as modeling

from config.staging import (
    MODEL_PARAMETER_RESULTS,
    MODEL_DATA_SET,
)
from config.research import (
    INFERENCE_DATA,
)


def start_inference(version: str = "v1",
    threshold: float = 0.5,
    ):
    selected_ds = "ds4"
    results_path = MODEL_PARAMETER_RESULTS

    pre_processing.preprocessing_inference_pipeline(
        INFERENCE_DATA,
        results_path,
        version,
        selected_ds,
    )

    modeling.model_inference_pipeline(
        MODEL_DATA_SET,
        results_path,
        version,
        selected_ds,
        threshold,
    )

    return {"mode": "inference", 
            "status": "completed",
            "best_model": f"inference_results/{version}/best_model_ds4.pkl",
            "final_infences": f"inference_results/{version}/final_inferences.csv",}

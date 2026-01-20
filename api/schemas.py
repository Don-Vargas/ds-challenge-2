from pydantic import BaseModel, Field
from typing import List, Dict, Any


class TrainingRequest(BaseModel):
    version: str = Field(default="v1", example="v2")
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)


class DriftRequest(BaseModel):
    file_path: str
    features: List[str]
    target: str
    alpha: float = 0.05
    return_data: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "file_path": "src/research/dataset/test/ds4.csv",
                "features": [
                    "efficiency_binning_quantile",
                    "eff_times_minutes_binning_quantile",
                    "scoring_impact_binning_quantile",
                    "eff_per_point_binning_quantile",
                    "eff_per_min_binning_quantile",
                    "points_binning_quantile"
                ],
                "target": "target",
                "alpha": 0.05,
                "return_data": True
            }
        }


class DriftResponse(BaseModel):
    drift_results: List[dict]
    reference_data: List[dict]
    drifted_data: List[dict]

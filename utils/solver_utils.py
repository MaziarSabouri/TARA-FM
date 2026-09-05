from typing import Dict, Optional


def build_solver_config(config: Dict, num_inference_steps: Optional[int]) -> Dict:
    solver_config = dict(config.get("solver_args", {}))
    solver_config.setdefault("method", "midpoint")
    if num_inference_steps:
        solver_config["time_points"] = num_inference_steps
        solver_config["step_size"] = 1.0 / num_inference_steps
    solver_config.setdefault("time_points", 10)
    solver_config.setdefault("step_size", 1.0 / solver_config["time_points"])
    return solver_config



### src/package_results.py
import os
import json
import pickle

def package_experiment_results(exp_name, model, metrics, config, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, exp_name)
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print(f"Results saved to {path}")


import os
import json
import pickle

def package_experiment_results(exp_name, model, metrics, config, output_dir="results"):
    # Saving all my results in a folder so I can find them later
    os.makedirs(output_dir, exist_ok=True)

    # I'm making a new folder for this expermient using the exp_name
    path = os.path.join(output_dir, exp_name)
    os.makedirs(path, exist_ok=True)

    # I'm saving the model here so I can reload it latter if I want
    with open(os.path.join(path, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    # I prefer storing metrics in JSON because it's easy to read 
    with open(os.path.join(path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # I also save the config so I remember how I set up the model
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print(f"Resluts saved to {path}")

import itertools
import os
import subprocess
import json



param_space = {
    
    "shots": [32],
    "n_iters": [10],
    "backbone": ["ViT-B/16", "ViT-L/14", "ViT-B/32"],
    "lr": [1e-3],
    "r": [4],
    "seed": [5, 42, 123]
}

param_combinations = list(itertools.product(*param_space.values()))
param_keys = list(param_space.keys())

print(param_combinations)

output_dir = "/home/codebase/Yinmi/BM_OCT/output/up_32shot/"
os.makedirs(output_dir, exist_ok=True)
results_path = os.path.join(output_dir, "results.json")
log_path = os.path.join(output_dir, "log.txt")


results = []


for idx, combination in enumerate(param_combinations):
    params = dict(zip(param_keys, combination))
    print(f"Running combination {idx + 1}/{len(param_combinations)}: {params}")

    print("in cmd")

    cmd = [
        "python", "main.py",
        f"--shots={params['shots']}",
        f"--n_iters={params['n_iters']}",
        f"--backbone={params['backbone']}",
        f"--lr={params['lr']}",
        f"--r={params['r']}",
        f"--seed={params['seed']}"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Combination {idx + 1} succeeded.")

        result_entry = {
            "params": params,
            "status": "success",
            "output": result.stdout
        }

    except subprocess.CalledProcessError as e:
        print(f"Combination {idx + 1} failed: {e.stderr}")

        result_entry = {
            "params": params,
            "status": "failure",
            "output": e.stderr
        }

    results.append(result_entry)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    with open(log_path, "a") as log_file:
        log_file.write(f"Combination {idx + 1}: {params}, Status: {result_entry['status']}\n")

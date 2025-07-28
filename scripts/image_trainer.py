#!/usr/bin/env python3
"""
Standalone Advanced Image Model Training Script for G.O.D Tournaments
"""

import argparse
import json
import os
import subprocess
import sys
import toml
import hashlib

# --- Configuration ---
# Add project root to python path to import modules
# This allows the script to be run from anywhere and still find the core modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Now that the path is set, we can import from core
from core import constants as cst
from core.config.config_handler import save_config_toml
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType

def get_model_path(base_model_name: str) -> str:
    """Gets the correct path for a model from the cache."""
    model_folder = base_model_name.replace("/", "--")
    cache_path = f"/cache/models/{model_folder}"
    if os.path.isdir(cache_path):
        # Check for a single safetensors file in the directory
        files = [f for f in os.listdir(cache_path) if f.endswith(".safetensors")]
        if len(files) == 1:
            model_path = os.path.join(cache_path, files[0])
            print(f"Found model file in cache: {model_path}", flush=True)
            return model_path
    print(f"Using model directory from cache: {cache_path}", flush=True)
    return cache_path

def create_config(args, tuning_profiles):
    """Creates a dynamic and adaptive training configuration."""
    print("Creating dynamic training config...", flush=True)

    # Determine which profile to use dynamically for A/B testing
    task_hash = int(hashlib.md5(args.task_id.encode()).hexdigest(), 16)
    use_champion_profile = task_hash % 2 == 0

    if args.model_type == ImageModelType.SDXL.value:
        template_path = "configs/base_diffusion_sdxl.toml"
        profile_name = "sdxl_champion" if use_champion_profile else "sdxl_explorer"
        default_profile_name = "sdxl_default"
    elif args.model_type == ImageModelType.FLUX.value:
        template_path = "configs/base_diffusion_flux.toml"
        profile_name = "flux_champion" if use_champion_profile else "flux_explorer"
        default_profile_name = "flux_default"
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    print(f"Dynamically selected tuning profile: {profile_name}", flush=True)
    profile = tuning_profiles.get(profile_name, tuning_profiles.get(default_profile_name, {}))

    with open(template_path, "r") as f:
        config = toml.load(f)

    # Apply parameters from the selected profile
    if args.model_type == ImageModelType.SDXL.value:
        config["learning_rate"] = profile.get("learning_rate", config.get("learning_rate"))
    elif args.model_type == ImageModelType.FLUX.value:
         config["unet_lr"] = profile.get("unet_lr", config.get("unet_lr"))

    config["network_dim"] = profile.get("network_dim", config.get("network_dim"))
    config["network_alpha"] = profile.get("network_alpha", config.get("network_alpha"))
    config["lr_scheduler"] = profile.get("lr_scheduler", config.get("lr_scheduler"))

    # Adaptive training steps based on duration
    if args.hours_to_complete and args.hours_to_complete > 0:
        target_steps = int(args.hours_to_complete * profile.get("target_steps_per_hour", 1000))
        max_steps = profile.get("max_possible_steps", 4000)
        config["max_train_steps"] = min(target_steps, max_steps)
        print(f"Adaptive training steps: {config['max_train_steps']} (based on {args.hours_to_complete} hours)", flush=True)
    else:
        config["max_train_steps"] = profile.get("max_possible_steps", 1600)
        print(f"WARNING: No duration provided. Using fallback steps: {config['max_train_steps']}", flush=True)
    
    # Set model and output paths according to tournament spec
    config["pretrained_model_name_or_path"] = get_model_path(args.model)
    config["train_data_dir"] = f"/dataset/images/{args.task_id}/img/"
    
    # Correct output path as per docs
    output_dir = f"/app/checkpoints/{args.task_id}/{args.expected_repo_name}"
    os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = output_dir

    # Save the final config to a temporary location inside the container
    final_config_path = f"/tmp/{args.task_id}.toml"
    save_config_toml(config, final_config_path)
    print(f"Created final config at {final_config_path}", flush=True)
    return final_config_path

def run_training(model_type, config_path):
    """Launches the training process using accelerate."""
    print(f"Starting training with config: {config_path}", flush=True)
    
    # Determine the correct training script based on model type
    train_script = "sdxl_train_network.py" if model_type == ImageModelType.SDXL.value else "flux_train_network.py"

    command = [
        "accelerate", "launch",
        "--num_processes=1",
        "--num_machines=1",
        f"/app/sd-scripts/{train_script}",
        f"--config_file={config_path}"
    ]
    
    print(f"Executing command: {' '.join(command)}", flush=True)
    
    # Stream output for real-time logging
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in iter(process.stdout.readline, ''):
        print(line, end='', flush=True)
    
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {process.returncode}")

def main():
    parser = argparse.ArgumentParser(description="G.O.D Tournament Image Training Script")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-zip", required=True)
    parser.add_argument("--model-type", required=True, choices=[e.value for e in ImageModelType])
    parser.add_argument("--hours-to-complete", type=float, default=1.0)
    parser.add_argument("--expected-repo-name", required=True)
    args = parser.parse_args()

    print("--- G.O.D ADVANCED IMAGE TRAINER ---", flush=True)
    print(f"Task ID: {args.task_id}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Model Type: {args.model_type}", flush=True)

    # Load tuning profiles from the configs directory
    with open("configs/tuning_profiles.json", "r") as f:
        tuning_profiles = json.load(f)

    # Create the dynamic config file for this run
    config_path = create_config(args, tuning_profiles)

    # Prepare the dataset
    print("Preparing dataset...", flush=True)
    dataset_zip_path = f"/cache/datasets/{args.task_id}.zip"
    if not os.path.exists(dataset_zip_path):
         raise FileNotFoundError(f"Dataset zip not found at {dataset_zip_path}. Ensure downloader ran successfully.")
    
    prepare_dataset(
        training_images_zip_path=dataset_zip_path,
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
    )

    # Run the training
    run_training(args.model_type, config_path)
    
    print("--- TRAINING SCRIPT COMPLETED SUCCESSFULLY ---", flush=True)

if __name__ == "__main__":
    main() 
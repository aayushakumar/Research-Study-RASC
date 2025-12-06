"""
RASC Paper Reproduction Script
This script systematically runs all core experiments from the RASC paper.
"""

import subprocess
import os
import sys
import time
from datetime import datetime

# Experiment configurations matching the paper
EXPERIMENTS = {
    "main_results": {
        "description": "Table 2: Main performance comparison (N=5, T=0.5)",
        "configs": [
            {"threshold": 0.5, "N": 5, "stop_mechanism": "PositiveN"}
        ]
    },
    "hyperparameter_study": {
        "description": "Figure 3: Hyperparameter analysis (varying N and T)",
        "configs": [
            # Varying threshold with N=5
            {"threshold": 0.1, "N": 5, "stop_mechanism": "PositiveN"},
            {"threshold": 0.2, "N": 5, "stop_mechanism": "PositiveN"},
            {"threshold": 0.3, "N": 5, "stop_mechanism": "PositiveN"},
            {"threshold": 0.4, "N": 5, "stop_mechanism": "PositiveN"},
            {"threshold": 0.5, "N": 5, "stop_mechanism": "PositiveN"},
            {"threshold": 0.6, "N": 5, "stop_mechanism": "PositiveN"},
            {"threshold": 0.7, "N": 5, "stop_mechanism": "PositiveN"},
            # Varying N with T=0.5
            {"threshold": 0.5, "N": 3, "stop_mechanism": "PositiveN"},
            {"threshold": 0.5, "N": 4, "stop_mechanism": "PositiveN"},
            {"threshold": 0.5, "N": 5, "stop_mechanism": "PositiveN"},
            {"threshold": 0.5, "N": 6, "stop_mechanism": "PositiveN"},
            {"threshold": 0.5, "N": 7, "stop_mechanism": "PositiveN"},
        ]
    },
    "stopping_mechanism": {
        "description": "Comparison of PositiveN vs ConsistencyN",
        "configs": [
            {"threshold": 0.5, "N": 5, "stop_mechanism": "PositiveN"},
            {"threshold": 0.5, "N": 5, "stop_mechanism": "ConsistencyN"}
        ]
    },
    "feature_ablation": {
        "description": "Table 4: Feature ablation study (answer vs reasoning vs combined)",
        "configs": [
            {"threshold": 0.5, "N": 5, "stop_mechanism": "PositiveN", "feature_type": "answer"},
            {"threshold": 0.5, "N": 5, "stop_mechanism": "PositiveN", "feature_type": "reasoning"},
            {"threshold": 0.5, "N": 5, "stop_mechanism": "PositiveN", "feature_type": "combined"}
        ]
    },
    "multi_model": {
        "description": "Table 2 Multi-Model: Performance across different base models",
        "configs": [
            {"threshold": 0.5, "N": 5, "stop_mechanism": "PositiveN", "model": "gpt-4"},
            {"threshold": 0.5, "N": 5, "stop_mechanism": "PositiveN", "model": "gpt-3.5-turbo-0125"},
            {"threshold": 0.5, "N": 5, "stop_mechanism": "PositiveN", "model": "claude-3-haiku-20240307"}
        ]
    }
}

def run_experiment(threshold, N, stop_mechanism, experiment_name, feature_type=None, model=None):
    """Run a single experiment configuration."""
    print(f"\n{'='*80}")
    print(f"Running: {experiment_name}")
    print(f"Configuration: threshold={threshold}, N={N}, stop_mechanism={stop_mechanism}")
    if feature_type:
        print(f"Feature type: {feature_type}")
    if model:
        print(f"Model: {model}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        # Build command with optional feature_type and model
        cmd_args = [sys.executable, "CS_based_early_stopping.py", 
                    str(threshold), str(N), stop_mechanism]
        if feature_type:
            cmd_args.append(feature_type)
        else:
            cmd_args.append('combined')  # default feature type
        if model:
            cmd_args.append(model)
        
        # Run the experiment
        result = subprocess.run(
            cmd_args,
            cwd="src",
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ SUCCESS (took {elapsed_time:.2f}s)")
            feature_suffix = f"_features_{feature_type}" if feature_type and feature_type != 'combined' else ""
            model_suffix = f"_model_{model}" if model else ""
            output_file = f"df_threshold_{threshold}_N_{N}_stop_{stop_mechanism}{feature_suffix}{model_suffix}.csv"
            print(f"  Output saved to: result/experiments_output/{output_file}")
            return True, elapsed_time, None
        else:
            print(f"✗ FAILED (after {elapsed_time:.2f}s)")
            print(f"Error: {result.stderr}")
            return False, elapsed_time, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT (exceeded 1 hour)")
        return False, 3600, "Timeout exceeded"
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        return False, time.time() - start_time, str(e)

def main():
    """Main experiment runner."""
    print("\n" + "="*80)
    print("RASC Paper Reproduction - Experiment Runner")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if we're in the correct directory
    if not os.path.exists("src/CS_based_early_stopping.py"):
        print("\n✗ ERROR: Cannot find src/CS_based_early_stopping.py")
        print("Please run this script from the RASC-main directory")
        sys.exit(1)
    
    # Check if input data exists
    if not os.path.exists("data/Evaluation_CoTs/Final_results/final_with_feature.json"):
        print("\n✗ ERROR: Cannot find input data file")
        print("Expected: data/Evaluation_CoTs/Final_results/final_with_feature.json")
        sys.exit(1)
    
    print("\n✓ Pre-flight checks passed")
    
    # Create output directory if it doesn't exist
    os.makedirs("result/experiments_output", exist_ok=True)
    
    # Track results
    results_log = []
    total_experiments = sum(len(exp["configs"]) for exp in EXPERIMENTS.values())
    completed = 0
    failed = 0
    
    print(f"\nTotal experiments to run: {total_experiments}")
    
    # Check if using subset (10% sample) and adjust time estimates
    using_subset = os.path.exists("data/Evaluation_CoTs/Final_results/final_with_feature_subset.json")
    time_multiplier = 0.1 if using_subset else 1.0  # 10x faster with subset
    
    if using_subset:
        print("\n✓ Using 10% subset dataset (655 samples) - experiments will be fast!")
    
    # Ask user which experiments to run
    print("\nExperiment groups:")
    print(f"1. Main results only (1 experiment) - ~{int(10*time_multiplier)} {'seconds' if using_subset else 'minutes'}")
    print(f"2. Main + Hyperparameter study (13 experiments) - ~{int(120*time_multiplier)} {'seconds (~2 min)' if using_subset else 'minutes (~2 hours)'}")
    print(f"3. Main + Hyperparameter + Feature ablation (16 experiments) - ~{int(160*time_multiplier)} {'seconds (~2.5 min)' if using_subset else 'minutes (~2.5 hours)'}")
    print(f"4. Main + Hyperparameter + Feature ablation + Multi-model (19 experiments) - ~{int(190*time_multiplier)} {'seconds (~3 min)' if using_subset else 'minutes (~3 hours)'}")
    print(f"5. All experiments (21 experiments) - ~{int(210*time_multiplier)} {'seconds (~3.5 min)' if using_subset else 'minutes (~3.5 hours)'}")
    print("6. Custom selection")
    
    choice = input("\nSelect option (1-6) [default: 1]: ").strip() or "1"
    
    experiments_to_run = []
    
    if choice == "1":
        experiments_to_run = ["main_results"]
    elif choice == "2":
        experiments_to_run = ["main_results", "hyperparameter_study"]
    elif choice == "3":
        experiments_to_run = ["main_results", "hyperparameter_study", "feature_ablation"]
    elif choice == "4":
        experiments_to_run = ["main_results", "hyperparameter_study", "feature_ablation", "multi_model"]
    elif choice == "5":
        experiments_to_run = list(EXPERIMENTS.keys())
    else:
        print("\nAvailable experiment groups:")
        for i, key in enumerate(EXPERIMENTS.keys(), 1):
            print(f"{i}. {key}: {EXPERIMENTS[key]['description']}")
        selected = input("Enter numbers separated by spaces (e.g., '1 2'): ").strip().split()
        experiment_keys = list(EXPERIMENTS.keys())
        experiments_to_run = [experiment_keys[int(i)-1] for i in selected if i.isdigit() and 0 < int(i) <= len(experiment_keys)]
    
    # Run selected experiments
    for exp_key in experiments_to_run:
        exp_data = EXPERIMENTS[exp_key]
        print(f"\n\n{'#'*80}")
        print(f"# {exp_data['description']}")
        print(f"{'#'*80}")
        
        for config in exp_data["configs"]:
            completed += 1
            exp_name = f"{exp_key}_{config['threshold']}_N{config['N']}_{config['stop_mechanism']}"
            if config.get('feature_type'):
                exp_name += f"_{config['feature_type']}"
            if config.get('model'):
                exp_name += f"_{config['model']}"
            
            print(f"\nProgress: {completed}/{total_experiments}")
            
            success, duration, error = run_experiment(
                config["threshold"],
                config["N"],
                config["stop_mechanism"],
                exp_name,
                config.get('feature_type'),  # Pass feature_type if present
                config.get('model')  # Pass model if present
            )
            
            results_log.append({
                "experiment": exp_name,
                "threshold": config["threshold"],
                "N": config["N"],
                "stop_mechanism": config["stop_mechanism"],
                "success": success,
                "duration": duration,
                "error": error
            })
            
            if not success:
                failed += 1
    
    # Print summary
    print("\n\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Total experiments: {completed}")
    print(f"Successful: {completed - failed}")
    print(f"Failed: {failed}")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save detailed log
    log_file = f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_file, 'w') as f:
        f.write("RASC Experiment Run Log\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for result in results_log:
            f.write(f"\n{'='*60}\n")
            f.write(f"Experiment: {result['experiment']}\n")
            f.write(f"Configuration: T={result['threshold']}, N={result['N']}, stop={result['stop_mechanism']}\n")
            f.write(f"Status: {'SUCCESS' if result['success'] else 'FAILED'}\n")
            f.write(f"Duration: {result['duration']:.2f}s\n")
            if result['error']:
                f.write(f"Error: {result['error']}\n")
    
    print(f"\nDetailed log saved to: {log_file}")
    
    if failed == 0:
        print("\n✓ All experiments completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python analyze.py' to generate comparison tables")
        print("2. Check result/experiments_output/ for individual experiment CSVs")
    else:
        print(f"\n⚠ {failed} experiment(s) failed. Check the log file for details.")

if __name__ == "__main__":
    main()

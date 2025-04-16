import os
import time
import random
import pandas as pd
from pathlib import Path
from ortools.algorithms.python import knapsack_solver

def parse_knapsack_file(filepath):
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        try:
            n = int(lines[0])
        except Exception as e:
            raise ValueError(f"Error parsing number of items in {filepath}: {e}")
        try:
            capacity = int(lines[1])
        except Exception as e:
            raise ValueError(f"Error parsing capacity in {filepath}: {e}")
            
        profits = []
        weight_list = []
        for line in lines[2:]:
            try:
                p, w = map(int, line.split())
            except Exception as e:
                raise ValueError(f"Error parsing profit and weight from line '{line}' in {filepath}: {e}")
            profits.append(p)
            weight_list.append(w)
        # For a single-dimension knapsack, weights must be a list containing one list with all weights.
        weights = [weight_list]

    return profits, weights, [capacity]

def solve_knapsack(profits, weights, capacities, time_limit_seconds=60):
    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "KnapsackExample"
    )
    solver.set_time_limit(time_limit_seconds)
    solver.init(profits, weights, capacities)
    
    start_time = time.time()
    computed_value = solver.solve()
    duration = time.time() - start_time
    if abs(duration - time_limit_seconds) <= 0.05:
        optimal = "no"
    else:
        optimal = "yes"
    packed_items = [i for i in range(len(profits)) if solver.best_solution_contains(i)]
    total_weight = sum(weights[0][i] for i in packed_items)

    return {
        'value': computed_value,
        'weight': total_weight,
        'items': packed_items,
        'is_optimal': optimal,
        'duration': duration
    }

def run_random_tests(kplib_root, result_path="knapsack_random_results.csv"):
    base_path = Path(kplib_root) / "kplib"
    target_sizes = [100, 200, 500]
    target_size_dirs = {"n" + str(size).zfill(5) for size in target_sizes}
    results = []

    for group in sorted(base_path.iterdir()):
        if not group.is_dir():
            continue
        print(f"\nProcessing group {group.name}")

        valid_files = []
        for size_dir in target_size_dirs:
            size_path = group / size_dir
            if not size_path.exists() or not size_path.is_dir():
                continue
            # Traverse each capacity folder (e.g., R01000, R10000, etc.)
            for capacity_folder in size_path.iterdir():
                if not capacity_folder.is_dir() or not capacity_folder.name.startswith("R"):
                    continue
                # Collect all .kp files in the capacity folder.
                files = list(capacity_folder.glob("*.kp"))
                valid_files.extend(files)

        if len(valid_files) < 5:
            print(f"Group {group.name}: found only {len(valid_files)} valid test cases. Skipping group.")
            continue

        selected_tests = random.sample(valid_files, 5)
        for kp_file in selected_tests:
            try:
                print(f"Doing test file {kp_file}")
                relative_path = kp_file.relative_to(base_path)
                print(f"  ➤ Solving {relative_path}...", end=' ')
                profits, weights, capacities = parse_knapsack_file(kp_file)
                result = solve_knapsack(profits, weights, capacities)

                results.append({
                    'group': group.name,
                    'file': str(relative_path),
                    'value': result['value'],
                    'total_weight': result['weight'],
                    'num_items': len(profits),
                    'num_packed': len(result['items']),
                    'is_optimal': result['is_optimal'],
                    'time_sec': round(result['duration'], 2)
                })
                print(f"Done in {result['duration']:.2f}s | Optimal: {result['is_optimal']}")
            except Exception as e:
                print(f"Error solving {relative_path}: {e}")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(result_path, index=False)
    print(f"\n All results saved to: {result_path}")

KPLIB_DIR = "./"
run_random_tests(KPLIB_DIR)

import subprocess
import sys
import numpy as np
import os
from itertools import product
import argparse

# Initialize argument parser
parser = argparse.ArgumentParser(description="Set the configuration for select_config.")
# Define the argument for select_config
parser.add_argument('--select_config', type=int, required=True, help="Pass the value for select_config")
# Parse arguments
args = parser.parse_args()
# Get the value passed for select_config
select_config = args.select_config
#################################################
if select_config == 1:
    # Config A: Baseline
    config_type = "Full_participation"
    num_rounds_list = [3, 5, 10, 20, 30, 50, 100, 200]      
    #num_rounds_list = [200]          
    num_partitions_list = [100]         
    fraction_fit_list = [1.0] 
    fraction_eval_list = [1.0] 
elif select_config == 2:
    # Config B: Partial Participation (Realistic Simulation)
    config_type = "Partial_participation"
    num_rounds_list = [3, 5, 10, 20, 30, 50, 100, 200]  
    #num_rounds_list = [200]               
    num_partitions_list = [100]         
    fraction_fit_list = [0.3] 
    fraction_eval_list = [0.3] 
elif select_config == 3:
    # Config C: Low Eval, High Train
    config_type = "LowEval_HighTrain"
    num_rounds_list = [3, 5, 10, 20, 30, 50, 100, 200]               
    #num_rounds_list = [200]  
    num_partitions_list = [100]         
    fraction_fit_list = [0.8] 
    fraction_eval_list = [0.2] 
elif select_config == 4:
    # Config D: High Eval, Low Train
    config_type = "HighEval_LowTrain"
    num_rounds_list = [3, 5, 10, 20, 30, 50, 100, 200]               
    #num_rounds_list = [200]  
    num_partitions_list = [100]         
    fraction_fit_list = [0.3] 
    fraction_eval_list = [0.9] 
elif select_config == 5:
    # Config E: Short Training Horizon
    config_type = "Short_Train"
    num_rounds_list = [3, 5, 10, 20, 30, 50, 100, 200]               
    #num_rounds_list = [200]  
    num_partitions_list = [100]         
    fraction_fit_list = [0.5] 
    fraction_eval_list = [0.5] 
else:
    print("wrong configuration selection")
    sys.exit(1)
    
#########################################################

fedavgcustom_path = f"./output/mnist_decentralized_fedavgcustom({config_type}_P{num_partitions_list[0]}).json"
performance_path = f"./output/mnist_decentralized_performance({config_type}_P{num_partitions_list[0]}).json"
result_path = f"./output/mnist_decentralized_result({config_type}_P{num_partitions_list[0]}).json"

#if os.path.exists(fedavgcustom_path):
#    os.remove(fedavgcustom_path)
#if os.path.exists(performance_path):
#    os.remove(performance_path)
#if os.path.exists(result_path):
#    os.remove(result_path)

# Output directory for experiment results
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Iterate through all combinations
combinations = product(num_rounds_list, num_partitions_list, fraction_fit_list, fraction_eval_list)

for num_rounds, num_partitions, frac_fit, frac_eval in combinations:
    output_name = f"R{num_rounds}_P{num_partitions}_F{frac_fit}_E{frac_eval}"
    output_path = os.path.join(output_dir, f"{output_name}.json")
    print("#"*50)
    print(f"Running: {output_name}")

    subprocess.run([
        "python", "mnist_decentralized.py",
        "--num_rounds", str(num_rounds),
        "--num_partitions", str(num_partitions),
        "--fraction_fit", str(frac_fit),
        "--fraction_evaluation", str(frac_eval),
        "--fedavgcustom_file", fedavgcustom_path,
        "--performance_file", performance_path,
        "--result_file", result_path,


    ])



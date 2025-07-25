import os
import subprocess

def generate_workload_for_solver(params):
    """
    Generates the workload trace files for a given set of parameters.
    This function is non-interactive and designed for the solver.
    """
    # The workload compiler is now expected to be at a different path
    workload_generator_path = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'extern', 'symbolic_tensor_graph', 'main.py'
    )
    
    # Ensure the temporary directory exists
    temp_dir = params["temp_dir"]
    os.makedirs(temp_dir, exist_ok=True)

    dp = params['dp']
    mp = params['tp'] # In the new format, mp is tp
    ssp = params['sp']
    pp = params['pp']
    
    # sharding is not fully implemented in the solver yet, defaulting to False
    sharded = False

    # Construct the command to run the workload generator
    cmd = (
        f"python {workload_generator_path} "
        f"--output_dir {temp_dir} "
        f"--output_name '{dp}_{mp}_{ssp}_{pp}_{1 if sharded else 0}.%d.et' "
        f"--comm_group '{dp}_{mp}_{ssp}_{pp}_{1 if sharded else 0}.json' "
        f"--dp {dp} "
        f"--tp {mp} "
        f"--sp {ssp} "
        f"--pp {pp} "
        f"--din {params['din']} "
        f"--dout {params['dout']} "
        f"--dmodel {params['dmodel']} "
        f"--dff {params['dff']} "
        f"--batch {params['batch']} "
        f"--seq {params['seq']} "
        f"--head {params['head']} "
        f"--num_stacks {params['num_stacks']} "
        f"--weight_sharded {sharded} "
        f"--chakra_schema_version v0.0.4"
    )

    print(cmd)
    # Run the workload generator
    process = subprocess.run(cmd, shell=True, cwd=None, capture_output=True, text=True)
    
    # Return True if the command was successful
    return process.returncode == 0

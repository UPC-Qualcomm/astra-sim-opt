import os
import sys
import subprocess

sys.path.append('/media/mohammad/extension/experiments/astra-sim/upc')
from generate_workloads import Model


def generate_workload_with_env(design_point, model, folder_name):
    """Generate workload using the correct Python environment"""
    root = os.path.join("/media/mohammad/extension/experiments/astra-sim/upc", "workload", folder_name)
    dp, mp, ssp, pp, sharded = design_point

    din, dout, dmodel, dff, batch, seq, head, num_stacks = Model.get_model_params(model)

    cmd = (
        f"/media/mohammad/extension/experiments/astraenv39/bin/python main.py "
        f"--output_dir {root} "
        f"--output_name {dp}_{mp}_{ssp}_{pp}_{1 if sharded else 0}.%d.et "
        f"--dp {dp} "
        f"--tp {mp} "
        f"--sp {ssp} "
        f"--pp {pp} "
        f"--dvocal {din} "
        f"--dmodel {dmodel} "
        f"--dff {dff} "
        f"--batch '{batch}' "
        f"--seq {seq} "
        f"--head {head} "
        f"--num_stacks {num_stacks} "
        f"--weight_sharded {sharded} "
        f"--chakra_schema_version v0.0.4"
    )
    cwd = "/media/mohammad/extension/experiments/astra-sim/extern/symbolic_tensor_graph"
    
    print(cmd)
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    return result.returncode == 0


def get_design_space(num_npus=64, dp_range=None, mp_range=None, pp_range=None, sharded_options=None):
    """Generate all possible parallelism configurations for given NPUs"""
    if dp_range is None:
        dp_range = [1, 2, 4, 8]
    if mp_range is None:
        mp_range = [1, 2, 4, 8, 16, 32, 64, 128]
    if pp_range is None:
        pp_range = [1, 2, 4, 8, 16, 32, 64]
    if sharded_options is None:
        sharded_options = [True, False]
    
    design_space = []
    
    for dp in dp_range:
        for mp in mp_range:
            for pp in pp_range:
                for sharded in sharded_options:
                    # Calculate spatial parallelism
                    sp = num_npus // (dp * mp * pp)
                    
                    # Check if configuration is valid
                    if sp >= 1 and dp * mp * sp * pp == num_npus:
                        design_space.append((dp, mp, sp, pp, sharded))
    
    return design_space
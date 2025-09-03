#!/usr/bin/python3
import os
import subprocess
from enum import Enum


def run_command(command, cwd=None):
    result = subprocess.run(command, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"run fail! {command}")
    return True


class Model(Enum):
    T5_Small = 0
    T5_Base = 1
    T5_Large = 2
    GPT_2_Small = 3
    GPT_2_Medium = 4
    GPT_3_1300M = 5
    GPT_Neo_2700M = 6
    FLAN_T5_XXL_11B = 7
    OPT_13B = 8
    GPT_NeoX_20B = 9
    GPT_3_175B = 10
    PaLM_540B = 11
    GPT_4_Estimated_over_1T = 12
    Default = 13

    @staticmethod
    def get_model_params(model):
        """Returns parameters as
        [din, dout, dmodel, dff, batch, seq, head, num_stacks]
        """
        if model == Model.T5_Small:
            return [32128, 512, 512, 2048, 64, 512, 8, 6]
        elif model == Model.T5_Base:
            return [32128, 768, 768, 3072, 32, 512, 12, 12]
        elif model == Model.T5_Large:
            return [32128, 1024, 1024, 4096, 16, 512, 16, 24]
        elif model == Model.GPT_2_Small:
            return [50257, 768, 768, 3072, 12, 1024, 12, 12]
        elif model == Model.GPT_2_Medium:
            return [50257, 1024, 1024, 4096, 8, 1024, 16, 24]
        elif model == Model.GPT_3_1300M:
            # return [50257, 2048, 2048, 8192, [1,2,4,8,16], 2048, 16, 2]
            return [50257, 2048, 2048, 8192, 4, 2048, 16, 24]
        elif model == Model.GPT_Neo_2700M:
            return [50257, 2560, 2560, 10240, 16, 2048, 32, 32]
        elif model == Model.FLAN_T5_XXL_11B:
            return [32128, 4096, 4096, 10240, 16, 512, 64, 24]
        elif model == Model.OPT_13B:
            return [50257, 5120, 5120, 20480, 8, 2048, 40, 40]
        elif model == Model.GPT_NeoX_20B:
            return [50257, 6144, 6144, 24576, 4, 2048, 64, 44]
        elif model == Model.GPT_3_175B:
            return [50257, 12288, 12288, 49152, 1, 2048, 96, 96]
        elif model == Model.PaLM_540B:
            return [50257, 18432, 18432, 73728, 1, 8192, 72, 118]
        elif model == Model.GPT_4_Estimated_over_1T:
            return [50257, 20480, 20480, 81920, 1, 8192, 128, 128]
        else:
            return [51200, 25600, 25600, 25600 * 4, 1024, 1024, 1024, 32]


model_display_names = {
    Model.T5_Small: "T5 Small",
    Model.T5_Base: "T5 Base",
    Model.T5_Large: "T5 Large",
    Model.GPT_2_Small: "GPT-2 Small",
    Model.GPT_2_Medium: "GPT-2 Medium",
    Model.GPT_3_1300M: "GPT-3 1300M",
    Model.GPT_Neo_2700M: "GPT Neo 2700M",
    Model.FLAN_T5_XXL_11B: "FLAN T5 XXL 11B",
    Model.OPT_13B: "OPT 13B",
    Model.GPT_NeoX_20B: "GPT NeoX 20B",
    Model.GPT_3_175B: "GPT-3 175B",
    Model.PaLM_540B: "PaLM 540B",
    Model.GPT_4_Estimated_over_1T: "GPT-4 Estimated over 1T",
    Model.Default: "Default",
}


def generate_trace(parallelism_strategy, parameters, temp_dir):
    root = os.path.join(os.path.split(os.path.abspath(__file__))[0], "../"+temp_dir)

    dp, mp, ssp, pp, sharded = parallelism_strategy
    din, dout, dmodel, dff, batch, seq, head, num_stacks = parameters
    cmd = (
        f"python main.py "
        f"--output_dir {root} "
        f"--output_name {dp}_{mp}_{ssp}_{pp}_{1 if sharded else 0}.%d.et "
        f"--comm_group {dp}_{mp}_{ssp}_{pp}_{1 if sharded else 0}.json "
        f"--dp {dp} "
        f"--tp {mp} "
        f"--sp {ssp} "
        f"--pp {pp} "
        f"--din {din} "
        f"--dout {dout} "
        f"--dmodel {dmodel} "
        f"--dff {dff} "
        f"--batch {batch} "  #'{batch}' "
        f"--seq {seq} "
        f"--head {head} "
        f"--num_stacks {num_stacks} "
        f"--weight_sharded {sharded} "
        f"--chakra_schema_version v0.0.4"
    )
    cwd = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        "../../..",
        "extern",
        "symbolic_tensor_graph",
    )
    #print(cmd)
    run_command(cmd, cwd)

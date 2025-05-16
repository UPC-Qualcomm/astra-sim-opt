#!/usr/bin/python3
import os
import subprocess
import multiprocessing
import argparse
from enum import Enum
from tqdm import tqdm


def run_command(command, cwd=None):
    result = subprocess.run(command, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"run fail! {command}")
    return True


def get_design_space(
    num_npus=64,
    dp={1, 2, 4, 8, 16},
    mp={1, 2, 4, 8, 16},
    pp={1, 2, 4, 8, 16},
    sharded={True, False},
):
    design_space = list()

    for ddp in dp:
        for mmp in mp:
            for ssharded in sharded:
                for ppp in pp:
                    ssp = num_npus // (ddp * mmp * ppp)
                    if ssp < 1:
                        continue
                    design_space.append((ddp, mmp, ssp, ppp, ssharded))
    return design_space

def get_design_space_no_sp(
    num_npus=64,
    dp={1, 2, 4, 8, 16},
    mp={1, 2, 4, 8, 16},
    pp={1, 2, 4, 8, 16},
    sharded={True, False},
):
    design_space = list()
    ssp=1
    for ddp in dp:
        for mmp in mp:
            for ssharded in sharded:
                for ppp in pp:
                    if num_npus != (ddp * mmp * ppp):
                        continue
                    design_space.append((ddp, mmp, ssp, ppp, ssharded))
    return design_space

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
        match model:
            case Model.T5_Small:
                return [32128, 512, 512, 2048, 64, 512, 8, 6]
            case Model.T5_Base:
                return [32128, 768, 768, 3072, 32, 512, 12, 12]
            case Model.T5_Large:
                return [32128, 1024, 1024, 4096, 16, 512, 16, 24]
            case Model.GPT_2_Small:
                return [50257, 768, 768, 3072, 12, 1024, 12, 12]
            case Model.GPT_2_Medium:
                return [50257, 1024, 1024, 4096, 8, 1024, 16, 24]
            case Model.GPT_3_1300M:
                #return [50257, 2048, 2048, 8192, [1,2,4,8,16], 2048, 16, 2]
                return [50257, 2048, 2048, 8192, 4, 2048, 16, 24]
            case Model.GPT_Neo_2700M:
                return [50257, 2560, 2560, 10240, 16, 2048, 32, 32]
            case Model.FLAN_T5_XXL_11B:
                return [32128, 4096, 4096, 10240, 16, 512, 64, 24]
            case Model.OPT_13B:
                return [50257, 5120, 5120, 20480, 8, 2048, 40, 40]
            case Model.GPT_NeoX_20B:
                return [50257, 6144, 6144, 24576, 4, 2048, 64, 44]
            case Model.GPT_3_175B:
                return [50257, 12288, 12288, 49152, 1, 2048, 96, 96]
            case Model.PaLM_540B:
                return [50257, 18432, 18432, 73728, 1, 8192, 72, 118]
            case Model.GPT_4_Estimated_over_1T:
                return [50257, 20480, 20480, 81920, 1, 8192, 128, 128]
            case _:
                return [51200, 25600, 25600, 25600 * 4, 1024, 1024, 1024, 32]

    # def get_model_params(model):
    #    din = 51200
    #    dout=25600
    #    dmodel=25600
    #    dff=25600*4
    #    batch=1024
    #    seq=1024
    #    head=1024
    #    num_stacks=32
    #    return [din, dout, dmodel, dff, batch, seq, head, num_stacks]


def generate_instance(design_point, model=Model.Default, folder_name="default"):
    root = os.path.join(
        os.path.split(os.path.abspath(__file__))[0], "workload", folder_name
    )
    dp, mp, ssp, pp, sharded = design_point

    din, dout, dmodel, dff, batch, seq, head, num_stacks = Model.get_model_params(model)

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
        f"--batch {batch} "#'{batch}' "
        f"--seq {seq} "
        f"--head {head} "
        f"--num_stacks {num_stacks} "
        f"--weight_sharded {sharded} "
        f"--chakra_schema_version v0.0.4"
    )
    cwd = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        "..",
        "extern",
        "symbolic_tensor_graph",
    )
    print(cmd)
    run_command(cmd, cwd)

if __name__ == "__main__":
    from functools import partial

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=int,
        help="The model to explore",
        required=False,
        default=Model.Default,
    )
    parser.add_argument(
        "--folder_name",
        type=str,
        help="The folder to dump generated files",
        required=False,
        default="Default",
    )
    args = parser.parse_args()

    num_npus = 64
    dp = {1, 2, 4, 8, 16}
    mp = {1, 2, 4, 8, 16}
    pp = {1, 2, 4, 8, 16}
    sharded = {True, False}
    model = args.model
    folder_name = args.folder_name

    design_space = get_design_space(num_npus, dp, mp, pp, sharded)
    #design_space = get_design_space_no_sp(num_npus, dp, mp, pp, sharded)
    func = partial(generate_instance, model=Model(int(model)), folder_name=folder_name)

    with multiprocessing.Pool(int(multiprocessing.cpu_count() * 0.95)) as pool:
        results = list(tqdm(pool.imap_unordered(func, design_space), total=len(design_space)))


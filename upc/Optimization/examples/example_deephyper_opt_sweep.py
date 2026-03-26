#!/usr/bin/env python3
"""
Dedicated DeepHyper sweep example.

This variant exists only to run the same optimization setup across multiple
objective functions without modifying `example_deephyper_opt.py`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Optimization import (
    create_search_space,
    RandomSampler,
    SimulationRunner,
    DeepHyperOptimizer,
    create_objective,
    get_available_objective_types,
)
from Optimization.core.base_optimizer import format_score


OBJECTIVE_METADATA = {
    "time": {"plot_labels": []},
    "time_and_network_bw": {"plot_labels": []},
    "power": {"plot_labels": []},
    "energy": {"plot_labels": []},
    "power_and_time": {"plot_labels": ["Total Power (W)", "Execution Cycles"]},
    "energy_and_time": {"plot_labels": ["Total Energy (J)", "Execution Cycles"]},
    "latency_total_network": {"plot_labels": ["Execution Cycles", "Network Total BW (GB/s)"]},
    "latency_network": {"plot_labels": ["log10(Execution Cycles)", "log10(Network Total BW (GB/s))"]},
    "latency_memory": {"plot_labels": ["log10(Execution Cycles)", "log10(Total Memory (GB))"]},
    "network_memory": {"plot_labels": ["log10(Network Total BW (GB/s))", "log10(Total Memory (GB))"]},
    "latency_network_memory": {"plot_labels": ["log10(Execution Cycles)", "log10(Network Total BW (GB/s))", "log10(Total Memory (GB))"]},
    "latency_network_raw": {"plot_labels": ["Execution Cycles", "Network Total BW (GB/s)"]},
    "latency_network_minmax": {"plot_labels": ["Normalized Time", "Normalized Network BW"]},
    "latency_network_sqrt": {"plot_labels": ["sqrt(Time)", "sqrt(Network BW)"]},
    "latency_network_power": {"plot_labels": ["Time^p", "Network BW^p"]},
    "edp": {"plot_labels": []},
    "edp_and_network_bw": {"plot_labels": ["EDP (J * cycles)", "Network Total BW (GB/s)"]},
    "ed2p_and_network_bw": {"plot_labels": ["ED²P (J * cycles²)", "Network Total BW (GB/s)"]},
    "e2d_and_network_bw": {"plot_labels": ["E²D (J² * cycles)", "Network Total BW (GB/s)"]},
    "energy_cycles_and_network_bw": {"plot_labels": ["Total Energy (J)", "Execution Cycles", "Network Total BW (GB/s)"]},
    "power_cycles_network_bw": {"plot_labels": ["Total Power (W)", "Execution Cycles", "Network Total BW (GB/s)"]},
    "ed2p": {"plot_labels": []},
    "e2d": {"plot_labels": []},
}

DEFAULT_OBJECTIVE = "e2d_and_network_bw"


def get_objective_key() -> str:
    available_objectives = get_available_objective_types()
    if len(sys.argv) > 1 and sys.argv[1] == "--list-objectives":
        print("\n".join(available_objectives))
        sys.exit(0)

    objective_key = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OBJECTIVE
    if objective_key not in available_objectives:
        available = ", ".join(sorted(available_objectives))
        raise ValueError(f"Unknown objective '{objective_key}'. Available: {available}")
    return objective_key


def main():
    objective_key = get_objective_key()
    objective_meta = OBJECTIVE_METADATA.get(objective_key, {"plot_labels": []})

    MODEL_NUM = 19
    MODEL_NAME = f"GPT_40B_{objective_key}"
    NUM_NPUS = 64
    NETWORK_NAME = "FoldedClos"
    BUDGET = 100
    INIT_SAMPLES = 20
    N_WORKERS = 8
    TOP_K = 10
    CLEANUP_BATCH_SIZE = 20
    COMPRESS_AND_CLEAN_IS_ENABLED = False
    
    print("=" * 70)
    print("EXAMPLE: DeepHyper Bayesian Optimization Sweep")
    print("=" * 70)
    print(f"Objective key: {objective_key}")
    print(f"Model: {MODEL_NAME}")
    print(f"NPUs: {NUM_NPUS}")
    print(f"Network: {NETWORK_NAME}")
    print(f"Budget: {BUDGET} evaluations")
    print(f"Workers: {N_WORKERS} (parallel evaluation)")
    print("Tracker: Enabled (kill at 1.5x threshold)\n")

    print("1. Creating search space...")
    search_space_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "search_space",
        "parallelism_strategy_params_g2_intra.json",
    )
    search_space = create_search_space(
        search_space_path,
        include_categories=["parallelism_strategy", "network"],
    )

    print("\n2. Creating sampler...")
    sampler = RandomSampler(seed=42)
    print(f"   Using: {sampler}")

    net_sim_config = {
        "sim_type": "g2",
        "topology": "FoldedClos",
        "paths_mode": "None",
        "routing_mode": "foldedclos_uniform",
        "estimate_power": 1,
        "power_config_path": os.path.join(os.environ["ASTRA_SIM_ROOT"], "upc", "power_model", "a100_config.json"),
        "topology_config": {
            "num_npus": search_space.num_npus,
            "npus_per_node": 8,
            "intra_node_topology": "switch",
            "num_nvswitches": 4,
            "bandwidth_config": {
                "host_edge": 100,
                "edge_agg": 100,
                "agg_core": 100,
                "intra_node": 450,
            },
            "bw_unit": "GB/s",
        },
    }

    print("\n3. Creating simulation runner...")
    sim_runner = SimulationRunner(
        model_num=MODEL_NUM,
        model_name=MODEL_NAME,
        network_name=NETWORK_NAME,
        folder_prefix="EXAMPLE_DEEPHYPER",
        verbose=True,
        net_sim_config=net_sim_config,
    )
    print(f"   Using: {sim_runner}")

    print("\n4. Creating objective function...")
    objective = create_objective(objective_type=objective_key)
    print(f"   Using: {objective.name}")

    plot_labels = objective_meta["plot_labels"]
    n_obj = len(plot_labels)
    moo_weight = [1.0 / n_obj] * n_obj if n_obj > 1 else None

    print("\n5. Creating DeepHyper optimizer...")
    optimizer = DeepHyperOptimizer(
        search_space=search_space,
        sampler=sampler,
        simulation_runner=sim_runner,
        budget=BUDGET,
        objective=objective,
        init_samples=INIT_SAMPLES,
        n_workers=N_WORKERS,
        acq_func="UCBd",
        acq_func_kwargs={"kappa": 10.0, "scheduler": {"type": "periodic-exp-decay", "period": 25, "kappa_final": 0.01}},
        surrogate_model="ET",
        surrogate_model_kwargs={"max_features": "sqrt"},
        acq_optimizer="mixedga",
        random_state=42,
        verbose=True,
        keep_top_k=TOP_K,
        profile_time=True,
        evaluator_method="process",
        acq_optimizer_kwargs={"max_total_failures": -1, "acq_optimizer_freq": 2},
        moo_scalarization_strategy="AugChebyshev",
        moo_scalarization_weight=moo_weight,
        enable_tracker=True,
        tracker_kill_multiplier=1.5,
        tracker_initial_threshold=1e15,
        cleanup_batch_size=CLEANUP_BATCH_SIZE,
        compress_and_clean_is_enabled=COMPRESS_AND_CLEAN_IS_ENABLED
    )
    print(f"   Using: {optimizer}")
    if optimizer.tracker:
        print(f"   Tracker: {optimizer.tracker}")

    print("\n" + "=" * 70)
    print("STARTING OPTIMIZATION")
    print("=" * 70)

    best_config, history = optimizer.run()

    if best_config is not None:
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)

        config_str = ", ".join([f"{k}={v}" for k, v in best_config.items()])
        print("\n🏆 BEST CONFIGURATION:")
        print(f"   {config_str}")
        print(f"   Score: {format_score(optimizer.best_score)}")
        print(f"\n📊 History saved with {len(history)} evaluations")

        if optimizer.tracker:
            print(optimizer.tracker)
            killed_count = history["was_killed"].sum() if "was_killed" in history.columns else 0
            kill_percentage = (killed_count / len(history) * 100) if len(history) > 0 else 0
            print(f"   Simulations killed: {killed_count}/{len(history)} ({kill_percentage:.1f}%)")
            if killed_count > 0:
                print(f"   ✅ Saved time by terminating {killed_count} slow simulation(s) early")

        print("\n💡 TIP: Check the objective-specific deephyper_results_*.csv for detailed DeepHyper output")

        if len(plot_labels) == 2:
            print("\n" + "=" * 70)
            print("GENERATING PLOTS")
            print("=" * 70)

            print("\n1. Plotting Pareto front (with outlier removal)...")
            try:
                from plot_pareto_front import plot_pareto_front

                csv_path = os.path.join(optimizer.save_dir, optimizer.results_filename)
                model_name = getattr(optimizer.simulation_runner, "model_name", "model")
                output_base = os.path.join(optimizer.save_dir, f"./pareto_front_{model_name}")

                pareto_plots = plot_pareto_front(
                    results_file=csv_path,
                    obj0_name=plot_labels[0],
                    obj1_name=plot_labels[1],
                    output_file=output_base,
                    plot_format="both",
                    show_labels=True,
                    remove_outliers=True,
                    iqr_multiplier=1.5,
                )
            except Exception as e:
                print(f"⚠️  Error plotting Pareto front: {e}")
                pareto_plots = None

            print("\n2. Plotting hypervolume indicator...")
            hv_path, hvi = optimizer.plot_hypervolume()

            if pareto_plots or hv_path:
                print("\n" + "=" * 70)
                print("VISUALIZATION COMPLETE")
                print("=" * 70)
                print("\n📈 Generated plots:")
                if pareto_plots:
                    for plot_path in pareto_plots:
                        if plot_path.endswith(".html"):
                            print(f"   - Pareto Front (Interactive): {plot_path}")
                        elif plot_path.endswith(".png"):
                            print(f"   - Pareto Front (Static): {plot_path}")
                if hv_path:
                    final_hvi = hvi[-1] if hasattr(hvi, "__len__") and len(hvi) > 0 else hvi
                    print(f"   - Hypervolume: {hv_path}")
                    print(f"   - Final HVI: {final_hvi:.4f}")
        elif len(plot_labels) > 2:
            print("\nℹ️  Skipping automatic plotting: this sweep example only auto-plots 2-objective runs.")
        else:
            print("\nℹ️  Skipping Pareto/hypervolume plots for single-objective runs.")
    else:
        print("\n❌ Optimization failed")


if __name__ == "__main__":
    main()

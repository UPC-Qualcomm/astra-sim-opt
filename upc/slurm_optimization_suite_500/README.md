# SLURM Optimization Suite

This bundle launches 12 DeepHyper optimization experiments on a SLURM cluster.

Models/objectives:
- llama-8b (32 NPUs): time, edp, energy_and_time
- gpt-60b (using model_num=19, 128 NPUs): time, edp, energy_and_time
- llama-70b (128 NPUs): time, edp, energy_and_time
- gpt-175b (1024 NPUs): time, edp, energy_and_time

## Run all at once

```bash
cd /scratch/nas/4/nasser/astra-sim/upc/slurm_optimization_suite_500
bash launch_all.sh
```

## Resource policy defaults

- CPU cores per job: 32% of assigned node cores
- Memory per CPU:
  - 4G when NPUs < 1024
  - 10G when NPUs >= 1024

You can override per experiment in `config.env`.

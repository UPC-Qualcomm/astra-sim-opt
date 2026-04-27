#!/bin/bash


python3 -m venv astraenv

source astraenv/bin/activate

pip3 install --upgrade pip

pip3 install protobuf==5.29.0

pip3 install graphviz pydot sympy tqdm seaborn matplotlib

pip3 install scikit-learn altair scipy umap-learn xgboost intervaltree ipykernel

cd ..

ASTRA_SIM=$(realpath ./astra-sim)

cd ${ASTRA_SIM}

git submodule update --init --recursive

# Chakra's ETFeederNode::get_chakra_node() is kept private upstream, but
# AstraSim's workload code needs to call it during the build; patch it here so
# the install remains self-contained and does not depend on a separate repo fix.
python3 - <<'PY'
from pathlib import Path

header = Path('extern/graph_frontend/chakra/src/feeder_v3/et_feeder_node.h')
lines = header.read_text().splitlines(keepends=True)

target_idx = -1
for i, line in enumerate(lines):
    if 'get_chakra_node() const;' in line and 'shared_ptr' in line:
        target_idx = i
        break

if target_idx == -1:
    raise SystemExit('Failed to find ETFeederNode::get_chakra_node() declaration')

# If the nearest previous non-empty line is already "public:", nothing to do.
prev_non_empty = target_idx - 1
while prev_non_empty >= 0 and lines[prev_non_empty].strip() == '':
    prev_non_empty -= 1

if prev_non_empty >= 0 and lines[prev_non_empty].strip() == 'public:':
    raise SystemExit(0)

indent = lines[target_idx][: len(lines[target_idx]) - len(lines[target_idx].lstrip())]
lines.insert(target_idx, f'{indent}public:\n')
header.write_text(''.join(lines))
PY

./build/astra_analytical/build.sh

ASTRA_SIM_BIN_AWARE=${ASTRA_SIM}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware
echo "export ASTRA_SIM_BIN_AWARE=${ASTRA_SIM_BIN_AWARE}" >> "${HOME}/.bashrc"
ASTRA_SIM_BIN_UNAWARE=${ASTRA_SIM}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware
echo "export ASTRA_SIM_BIN_UNAWARE=${ASTRA_SIM_BIN_UNAWARE}" >> "${HOME}/.bashrc"

echo "export ASTRA_SIM_ROOT=${ASTRA_SIM}" >> "${HOME}/.bashrc"
ASTRA_SIM_PYTHON=$(realpath ../astraenv/bin/python)
echo "export ASTRA_SIM_PYTHON=${ASTRA_SIM_PYTHON}" >> "${HOME}/.bashrc"

./build/astra_ns3/build.sh -c

cd ./extern/graph_frontend/chakra/

pip3 install .

cd ${ASTRA_SIM}
pip3 install -r ./requirements.txt

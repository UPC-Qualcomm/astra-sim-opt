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
import re

header = Path('extern/graph_frontend/chakra/src/feeder_v3/et_feeder_node.h')
text = header.read_text()
pattern = r'(\n\s*)private:\n(\s*)std::shared_ptr<const ChakraNode> get_chakra_node\(\) const;\n'
replacement = r'\1public:\n\2std::shared_ptr<const ChakraNode> get_chakra_node() const;\n'
updated_text, count = re.subn(pattern, replacement, text, count=1)
if count == 0:
    raise SystemExit('Failed to patch ETFeederNode::get_chakra_node() access level')
header.write_text(updated_text)
PY

./build/astra_analytical/build.sh

ASTRA_SIM_BIN_AWARE=${ASTRA_SIM}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware
echo "export ASTRA_SIM_BIN_AWARE=${ASTRA_SIM_BIN_AWARE}" >> "${HOME}/.bashrc"
ASTRA_SIM_BIN_UNAWARE=${ASTRA_SIM}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware
echo "export ASTRA_SIM_BIN_UNAWARE=${ASTRA_SIM_BIN_UNAWARE}" >> "${HOME}/.bashrc"
G2_SIM_BIN=${ASTRA_SIM}/build/astra_g2/build/bin/AstraSim_G2_congestion
echo "export G2_SIM_BIN=${G2_SIM_BIN}" >> "${HOME}/.bashrc"

echo "export ASTRA_SIM_ROOT=${ASTRA_SIM}" >> "${HOME}/.bashrc"
ASTRA_SIM_PYTHON=$(realpath ../astraenv39/bin/python)
echo "export ASTRA_SIM_PYTHON=${ASTRA_SIM_PYTHON}" >> "${HOME}/.bashrc"

./build/astra_ns3/build.sh -c

cd ./extern/graph_frontend/chakra/

pip3 install .

cd ${ASTRA_SIM}
pip3 install -r ./requirements.txt

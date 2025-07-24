#!/bin/bash


python -m venv astraenv

source astraenv/bin/activate

pip3 install --upgrade pip

pip3 install protobuf==5.29.0

pip3 install graphviz pydot sympy tqdm seaborn matplotlib

cd ..

ASTRA_SIM=$(realpath ./astra-sim)

cd ${ASTRA_SIM}

git submodule update --init --recursive

./build/astra_analytical/build.sh

ASTRA_SIM_BIN=${ASTRA_SIM}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware
echo "export ASTRA_SIM_BIN=${ASTRA_SIM_BIN}" >> "${HOME}/.bashrc"

./build/astra_ns3/build.sh -c

cd ./extern/graph_frontend/chakra/

pip3 install .
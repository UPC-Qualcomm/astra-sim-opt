#!/bin/bash


python -m venv astraenv

source astraenv/bin/activate

pip3 install --upgrade pip

pip3 install protobuf==5.29.0

pip3 install graphviz pydot matplotlib seaborn sympy

cd ..

ASTRA_SIM=$(realpath ./astra-sim)

cd ${ASTRA_SIM}

git submodule update --init --recursive

./build/astra_analytical/build.sh

ASTRA_SIM_BIN=${ASTRA_SIM}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware

./build/astra_ns3/build.sh -c

cd ./extern/graph_frontend/chakra/

pip3 install .
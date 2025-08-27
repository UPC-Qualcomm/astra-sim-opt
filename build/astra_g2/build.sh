#!/bin/bash
set -e

# set paths
SCRIPT_DIR=$(dirname "$(realpath "$0")")
CHAKRA_ET_DIR="${SCRIPT_DIR:?}"/../../extern/graph_frontend/chakra/schema/protobuf

# set functions
function compile_chakra_et() {
  # compile et_def.proto if one doesn't exist
  if [[ ! -f "${CHAKRA_ET_DIR:?}"/et_def.pb.h || ! -f "${CHAKRA_ET_DIR:?}"/et_def.pb.cc ]]; then
    protoc et_def.proto \
      --proto_path="${CHAKRA_ET_DIR:?}" \
      --cpp_out="${CHAKRA_ET_DIR:?}"
  fi

  if [[ ! -f "${CHAKRA_ET_DIR:?}"/et_def_pb2.py ]]; then
    protoc et_def.proto \
      --proto_path="${CHAKRA_ET_DIR:?}" \
      --python_out="${CHAKRA_ET_DIR:?}"
  fi
}

function setup() {
  # make build directory if one doesn't exist
  if [[ ! -d "${BUILD_DIR:?}" ]]; then
    mkdir -p "${BUILD_DIR:?}"
  fi

  # set concurrent build threads, capped at 16
  NUM_THREADS=$(nproc)
  if [[ ${NUM_THREADS} -ge 16 ]]; then
    NUM_THREADS=16
  fi
}

function compile_astrasim_g2() {
  # compile AstraSim
  cd "${BUILD_DIR:?}" || exit
  cmake .. -DBUILDTARGET="$1"
  cmake --build . -j "${NUM_THREADS:?}"
}

function compile_astrasim_g2_as_debug() {
  # compile AstraSim
  cd "${BUILD_DIR:?}" || exit
  cmake .. -DBUILDTARGET="$1" -DCMAKE_BUILD_TYPE=Debug
  cmake --build . --config=Debug -j "${NUM_THREADS:?}"
}

function cleanup() {
  rm -rf "${BUILD_DIR:?}"
  rm -f "${CHAKRA_ET_DIR}/et_def.pb.cc"
  rm -f "${CHAKRA_ET_DIR}/et_def.pb.h"
  rm -f "${CHAKRA_ET_DIR}/et_def_pb2.py"
}

function create_symlink_astrasim() {
  # create symlinks for backward compatibility
  # ASTRA-sim library
  if [[ ! -d "${BUILD_DIR:?}"/AstraSim/lib/ ]]; then
    mkdir -p "${BUILD_DIR:?}"/AstraSim/lib/
  fi

  if [[ ! -L "${BUILD_DIR:?}"/AstraSim/lib/libAstraSim.a ]]; then
    ln -s "${BUILD_DIR:?}"/lib/libAstraSim.a "${BUILD_DIR:?}"/AstraSim/lib/libAstraSim.a
  fi
}

function create_symlink_g2() {
  # create symlinks for backward compatibility
  # congestion_unaware
  if [[ ! -d "${BUILD_DIR:?}"/G2Astra/bin/ ]]; then
    mkdir -p "${BUILD_DIR:?}"/G2Astra/bin/
  fi

  if [[ ! -L "${BUILD_DIR:?}"/G2Astra/bin/G2Astra ]]; then
    ln -s "${BUILD_DIR:?}"/bin/AstraSim_G2 "${BUILD_DIR:?}"/G2Astra/bin/G2
  fi
}


function print_usage() {
  echo "print usage"
}

# set default option values
build_target="g2"
build_as_debug=false
should_clean=false

# Process command-line options
while getopts "t:ld" OPT; do
  case "${OPT:?}" in
  t)
    build_target="${OPTARG:?}"
    ;;
  l)
    should_clean=true
    ;;
  d)
    build_as_debug=true
    ;;
  *)
    exit 1
    ;;
  esac
done

# set build directory
if [[ ${build_as_debug:?} == true ]]; then
  BUILD_DIR="${SCRIPT_DIR:?}"/build_debug
else
  BUILD_DIR="${SCRIPT_DIR:?}"/build
fi

# check the validity of build target
if [[ ${build_target:?} != "g2" ]]; then
  echo "Invalid build target: ${build_target:?}" >&2
  exit 1
fi

# run operations as required
if [[ ${should_clean:?} == true ]]; then
  cleanup
else
  # setup ASTRA-sim build
  setup
  compile_chakra_et

  # compile ASTRA-sim
  if [[ ${build_as_debug:?} == true ]]; then
    compile_astrasim_g2_as_debug "${build_target:?}"
  else
    compile_astrasim_g2 "${build_target:?}"
  fi

  # create symlinks as appropriate (for backward compatibility)
  if [[ ${build_target:?} == "g2" ]]; then
    create_symlink_g2
  fi
fi

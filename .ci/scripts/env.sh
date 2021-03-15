#!/bin/bash -eEx

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"

# shellcheck disable=SC2034
DLRM_MODEL="small"

HOSTNAME=$(hostname -s)
export HOSTNAME
SRC_ROOT_DIR=$(cd "${SCRIPT_DIR}/../../" && pwd -P)
CONFIGS_DIR="${SRC_ROOT_DIR}/.ci/configs"
export HOSTFILE=${CONFIGS_DIR}/$HOSTNAME/hostfile.txt
# shellcheck disable=SC2002
HOSTS=$(cat "$HOSTFILE" | xargs | tr ' ' ',')
export HOSTS
NP=$(wc --lines "$HOSTFILE" | awk '{print $1}')
export NP

if [ ! -f "${HOSTFILE}" ]; then
    echo "ERROR: ${HOSTFILE} does not exist"
    exit 1
fi

export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=4242

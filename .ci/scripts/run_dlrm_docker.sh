#!/bin/bash -eEx
set -o pipefail

function err_report () {
    echo "Exited with ERROR in line $1"
    exit 1
}
trap 'err_report $LINENO' ERR

SCRIPT_DIR="$(
    cd "$(dirname "$0")"
    pwd -P
)"
cd "${SCRIPT_DIR}"
. "${SCRIPT_DIR}/env.sh"

TORCH_UCC_MODE="$1"

if [ "${TORCH_UCC_MODE}" != "ucc" ] && [ "${TORCH_UCC_MODE}" != "xccl" ]; then
    echo "ERROR: unsupported or empty TORCH_UCC_MODE (${TORCH_UCC_MODE}), supported values: ucc, xccl"
    exit 1
fi

export HOSTFILE=${HOSTFILE:-${CONFIGS_DIR}/$HOSTNAME/hostfile.txt}

if [ ! -f "${HOSTFILE}" ]; then
    echo "ERROR: ${HOSTFILE} does not exist"
    exit 1
fi

# shellcheck disable=SC2002
HOSTS=$(cat "$HOSTFILE" | xargs | tr ' ' ',')
export HOSTS
HEAD_NODE=$(head -1 "$HOSTFILE")
export HEAD_NODE

DOCKER_CONTAINER_NAME="torch_ucc"
# TODO debug
DOCKER_IMAGE_NAME="${TORCH_UCC_DOCKER_IMAGE_NAME}:${BUILD_ID}"
#DOCKER_IMAGE_NAME="harbor.mellanox.com/torch-ucc/1.0.0/x86_64/centos8/cuda11.2.1:205"

DOCKER_RUN_ARGS="\
--pull always \
--network=host \
--uts=host \
--ipc=host \
--ulimit stack=67108864 \
--ulimit memlock=-1 \
--security-opt seccomp=unconfined \
--cap-add=SYS_ADMIN \
--device=/dev/infiniband/ \
--gpus all \
--user root \
-it \
-d \
--rm \
--name=${DOCKER_CONTAINER_NAME} \
-v /labhome:/labhome \
-v /root/.ssh:/root/.ssh \
"

# shellcheck disable=SC2013
for HOST in $(cat "$HOSTFILE"); do
    echo "INFO: HOST = $HOST"

    STALE_DOCKER_CONTAINER_LIST=$(sudo ssh -n "$HOST" "docker ps -a -q -f name=${DOCKER_CONTAINER_NAME}")
    if [ -n "${STALE_DOCKER_CONTAINER_LIST}" ]; then
        echo "WARNING: stale docker container (name: ${DOCKER_CONTAINER_NAME}) is detected on ${HOST} (to be stopped)"
        echo "INFO: Stopping stale docker container (name: ${DOCKER_CONTAINER_NAME}) on ${HOST}..."
        sudo ssh "${HOST}" docker stop ${DOCKER_CONTAINER_NAME}
        echo "INFO: Stopping stale docker container (name: ${DOCKER_CONTAINER_NAME}) on ${HOST}... DONE"
    fi

    echo "INFO: start docker container on $HOST ..."
    # shellcheck disable=SC2029
    sudo ssh "$HOST" "docker run \
        ${DOCKER_RUN_ARGS} \
        ${DOCKER_IMAGE_NAME} \
        bash -c '/usr/sbin/sshd -p ${DOCKER_SSH_PORT}; sleep infinity'"
    echo "INFO: start docker container on $HOST ... DONE"

    sleep 5

    echo "INFO: verify docker container on $HOST ..."
    sudo ssh -p "${DOCKER_SSH_PORT}" "$HOST" hostname
    sudo ssh -p "${DOCKER_SSH_PORT}" "$HOST" cat /proc/1/cgroup
    echo "INFO: verify docker container on $HOST ... DONE"
done

# TODO remove sudo
sudo ssh -p "${DOCKER_SSH_PORT}" "${HEAD_NODE}" /opt/nvidia/torch-ucc/src/.ci/scripts/run_dlrm.sh ${TORCH_UCC_MODE} cpu /opt/nvidia/torch-ucc/src/.ci/configs/$HOSTNAME/hostfile.txt
sudo ssh -p "${DOCKER_SSH_PORT}" "${HEAD_NODE}" /opt/nvidia/torch-ucc/src/.ci/scripts/run_dlrm.sh ${TORCH_UCC_MODE} gpu /opt/nvidia/torch-ucc/src/.ci/configs/$HOSTNAME/hostfile.txt

# TODO debug
# shellcheck disable=SC2013
#for HOST in $(cat "$HOSTFILE"); do
#    echo "INFO: stop docker container on $HOST ..."
#    ssh "${HOST}" docker stop ${DOCKER_CONTAINER_NAME}
#    echo "INFO: stop docker container on $HOST ... DONE"
#done

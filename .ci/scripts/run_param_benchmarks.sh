#!/bin/bash -eEx
set -o pipefail

# TODO debug
exit 0

source /workspace/set-env-dist.sh
index=$LOCAL_RANK
export OMPI_COMM_WORLD_SIZE=$WORLD_SIZE
export OMPI_COMM_WORLD_LOCAL_SIZE=$LOCAL_SIZE
export OMPI_COMM_WORLD_RANK=$RANK
export OMPI_COMM_WORLD_LOCAL_RANK=$LOCAL_RANK

if (( $index == 0 )); then
    export UCX_NET_DEVICES=mlx5_0:1
    NUMA="numactl --physcpubind=48-63 --membind=3 "
elif (( $index == 1 )); then
    export UCX_NET_DEVICES=mlx5_1:1
    NUMA="numactl --physcpubind=48-63 --membind=3 "
elif (( $index == 2 )); then
    export UCX_NET_DEVICES=mlx5_2:1
    NUMA="numactl --physcpubind=16-31 --membind=1 "
elif (( $index == 3 )); then
    export UCX_NET_DEVICES=mlx5_3:1
    NUMA="numactl --physcpubind=16-31 --membind=1 "
elif (( $index == 4 )); then
    export UCX_NET_DEVICES=mlx5_6:1
    NUMA="numactl --physcpubind=112-127 --membind=7 "
elif (( $index == 5 )); then
    export UCX_NET_DEVICES=mlx5_7:1
    NUMA="numactl --physcpubind=112-127 --membind=7 "
elif (( $index == 6 )); then
    export UCX_NET_DEVICES=mlx5_8:1
    NUMA="numactl --physcpubind=80-95 --membind=5 "
elif (( $index == 7 )); then
    export UCX_NET_DEVICES=mlx5_9:1
    NUMA="numactl --physcpubind=80-95 --membind=5 "
fi

export XCCL_TEAM_UCX_NET_DEVICES=$UCX_NET_DEVICES
export XCCL_TEAM_HIER_NET_DEVICES=$UCX_NET_DEVICES

EXE="$NUMA python /workspace/param/train/comms/pt/comms.py \
            --master-ip $MASTER_ADDR \
            --master-port $MASTER_PORT $@"
$EXE

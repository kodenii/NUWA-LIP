#!/bin/bash
CHECKPOINT=""
while [[ "$#" -gt 0 ]]; do
  case $1 in
  -cf | --config)
    CF="$2"
    shift
    ;;
  -a | --action)
    ACTION="$2"
    shift
    ;;
  -ltbs | --local_train_batch_size)
    TRAIN_LOCAL_BATCH_SIZE="$2"
    shift
    ;;
  -lebs | --local_eval_batch_size)
    EVAL_LOCAL_BATCH_SIZE="$2"
    shift
    ;;
  -m | --mode)
    MODE="$2"
    shift
    ;;
  -p | --platform)
    PLATFORM="$2"
    shift
    ;;
  -d | --debug)
    DEBUG="$2"
    shift
    ;;
  # should be the format: node-0:0,node-1:1,...
  -crd | --node_rank_dict)
    CUSTOM_RANK_DICT="$2"
    shift
    ;;
  -crd_alt | --node_rank_dict_alt)
    CUSTOM_RANK_DICT_ALT="$2"
    shift
    ;;
  -cmp | --master_port)
    CUSTOM_MASTER_PORT="$2"
    shift
    ;;
  -ckpt | --checkpoint)
    CHECKPOINT="$2"
    shift
    ;;
  *)
    echo "Unknown parameter passed: $1"
    exit 1
    ;;
  esac
  shift
done
echo "====Input Params====="
echo "CF: $CF"
echo "ACTION: $ACTION"
echo "TRAIN_LOCAL_BATCH_SIZE: $TRAIN_LOCAL_BATCH_SIZE"
echo "EVAL_LOCAL_BATCH_SIZE: $EVAL_LOCAL_BATCH_SIZE"
echo "PLATFORM: $PLATFORM"
echo "DEBUG: $DEBUG"
echo "CHECKPOINT: $CHECKPOINT"

function mfcb { local val="$4"; "$1"; eval "$2[$3]=\$val;"; };
function val_ltrim { if [[ "$val" =~ ^[[:space:]]+ ]]; then val="${val:${#BASH_REMATCH[0]}}"; fi; };
function val_rtrim { if [[ "$val" =~ [[:space:]]+$ ]]; then val="${val:0:${#val}-${#BASH_REMATCH[0]}}"; fi; };
function val_trim { val_ltrim; val_rtrim; };



# kill Jobs
to_kill_job=$(ps -aux | grep "finetune.py" | grep -v grep| awk '{print $2}')
if [ ! $to_kill_job ]; then
  echo 'No finetune.py jobs to kill'
else
  echo $to_kill_job | xargs kill -9
fi

if [ ! $to_kill_job ]; then
  echo 'No finetune.py jobs to kill'
else
  echo 'Pls manually kill finetune related jobs!!!!!!!!!!!!!!!'
  exit 0
fi

#CF='config/g2v/SMS10000.py'
#NODE_COUNT=1
#LOCAL_BATCH_SIZE=6
if [ "$PLATFORM" = "sin" ]; then
  USED_NODE_COUNT=$NODE_COUNT
  USED_GPU_PER_NODE=$GPU_PER_NODE_COUNT
  USED_MASTER_ADDR=$MASTER_ADDR
  USED_MASTER_PORT=$MASTER_PORT
  USED_NODE_RANK=$NODE_RANK

  export LD_PRELOAD=/opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so
  export LD_LIBRARY_PATH=/opt/hpcx/nccl_rdma_sharp_plugin/lib/:$LD_LIBRARY_PATH
  export UCX_TLS=tcp
  export UCX_NET_DEVICES=eth0
  export UCX_IB_ENABLE_CUDA_AFFINITY=n
  export UCX_IB_PCI_RELAXED_ORDERING=on
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export NCCL_SOCKET_IFNAME=eth0
  export NCCL_NET_GDR_LEVEL=5
  export NCCL_IB_PCI_RELAXED_ORDERING=1
  export NCCL_TOPO_FILE=/opt/microsoft/ndv4-topo.xml
  export NCCL_DEBUG=WARN
  export NCCL_DEBUG_SUBSYS=INIT
  export PATH=$HOME/conda/bin:$PATH

elif [ "$PLATFORM" = "pai" ]; then
  USED_NODE_COUNT=$PAI_TASK_ROLE_TASK_COUNT_worker
  USED_GPU_PER_NODE=8
  USED_MASTER_ADDR=$PAI_HOST_IP_worker_0
  USED_MASTER_PORT="6002"
  USED_NODE_RANK=$PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX
elif [ "$PLATFORM" = "itp" ]; then
  USED_NODE_COUNT=$AZUREML_NODE_COUNT
  USED_GPU_PER_NODE=$DLWS_NUM_GPU_PER_WORKER
  USED_MASTER_ADDR=$MASTER_IP
  USED_MASTER_PORT="6002"
  USED_NODE_RANK=$NODE_RANK
elif [ "$PLATFORM" = "local" ]; then
  USED_NODE_COUNT=1
  USED_GPU_PER_NODE=`nvidia-smi --list-gpus | wc -l`
  USED_MASTER_ADDR=
  USED_MASTER_PORT=
  USED_NODE_RANK=
elif [ "$PLATFORM" = "custom" ]; then
  readarray -c1 -C 'mfcb val_trim a' -td, <<<"$CUSTOM_RANK_DICT,"; unset 'a[-1]';
  USED_NODE_COUNT=${#a[@]}
  USED_GPU_PER_NODE=`nvidia-smi --list-gpus | wc -l`
  USED_MASTER_ADDR=${a[0]}
  USED_MASTER_PORT=$CUSTOM_MASTER_PORT
  if [ "$CUSTOM_RANK_DICT_ALT" != "" ]; then
    readarray -c1 -C 'mfcb val_trim a' -td, <<<"$CUSTOM_RANK_DICT_ALT,"; unset 'a[-1]';
  fi
  for i in "${!a[@]}"; do
   if [[ "${a[$i]}" = "${HOSTNAME}" ]]; then
       USED_NODE_RANK="${i}";
   fi
  done
else
  echo "Invalid platform name: $PLATFORM"
  exit 1
fi

TRAIN_BATCH_SIZE=$(expr $USED_GPU_PER_NODE \* $USED_NODE_COUNT \* $TRAIN_LOCAL_BATCH_SIZE)
EVAL_BATCH_SIZE=$(expr $USED_GPU_PER_NODE \* $USED_NODE_COUNT \* $EVAL_LOCAL_BATCH_SIZE)

echo "====Using $PLATFORM platform====="
echo "USED_NODE_COUNT: $USED_NODE_COUNT"
echo "USED_GPU_PER_NODE: $USED_GPU_PER_NODE"
echo "USED_MASTER_ADDR: $USED_MASTER_ADDR"
echo "USED_MASTER_PORT: $USED_MASTER_PORT"
echo "USED_NODE_RANK: $USED_NODE_RANK"
echo "TRAIN_BATCH_SIZE: $TRAIN_BATCH_SIZE"
echo "EVAL_BATCH_SIZE: $EVAL_BATCH_SIZE"


DISTRIBUTED_ARGS="--nproc_per_node $USED_GPU_PER_NODE"
if [ "$PLATFORM" != "local" ]; then
  DISTRIBUTED_ARGS="$DISTRIBUTED_ARGS  \
                  --nnodes $USED_NODE_COUNT \
                  --node_rank $USED_NODE_RANK \
                  --master_addr $USED_MASTER_ADDR \
                  --master_port $USED_MASTER_PORT"
fi

MODEL_ARGS="--$MODE \
          --cf $CF \
          --$ACTION \
          --tbs $TRAIN_BATCH_SIZE \
          --ebs $EVAL_BATCH_SIZE"
 
if [ "$DEBUG" = 1 ]; then
  MODEL_ARGS="$MODEL_ARGS --debug"
fi
if [ "$CHECKPOINT" != "" ]; then 
  MODEL_ARGS="$MODEL_ARGS --ckpt $CHECKPOINT"
fi
echo "Running bash command:"
echo "python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune.py $MODEL_ARGS"
python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune.py $MODEL_ARGS

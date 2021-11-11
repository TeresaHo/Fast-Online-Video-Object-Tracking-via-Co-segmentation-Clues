


ROOT=`git rev-parse --show-toplevel`
export PYTHONPATH=$ROOT:$PYTHONPATH

mkdir -p logs

base=$1
saveD=$2

python -u $ROOT/tools/train_siammask_refine.py \
    --config=config.json -b 4 \
    -j 20  --pretrained $base \
    --epochs 10 \
    --proposals 100\
    --save_dir $saveD
    2>&1 | tee logs/train.log

GPUNAME=$1
if [[ -z $GPUNAME ]];
then
    echo `date`" - Missing mandatory arguments: GPU name. "
    exit 1
fi
export CUDA_VISIBLE_DEVICES=$1GPUNAME
rm -rf example_data/active/*
rm .activecache.json
python scripts/run_pseudo_al.py \
-di='example_data/images/' \
-lf='example_data/labels.csv' \
-df='example_data/_megadetector.json' \
-da='example_data/active/'

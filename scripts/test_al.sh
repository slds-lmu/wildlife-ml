export CUDA_VISIBLE_DEVICES=$1
rm -rf example_data/active/*
rm .activecache.json
python test_al.py \
-di='example_data/images/' \
-lf='example_data/labels.csv' \
-df='example_data/_megadetector.json' \
-da='example_data/active/'

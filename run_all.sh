#!/usr/bin/env bash
# Full re-run: simulator -> three pipelines -> figures. About 10-25 minutes per data source on 2 CPU cores.
set -e
cd "$(dirname "$0")"
python3 sim.py 80                    # writes leach_sim.csv (80 networks, seeds 1000..1079)
python3 run2.py leach > logs/log2_leach.txt 2>&1
python3 run2.py nsl   > logs/log2_nsl.txt   2>&1   # needs data/KDDTrain.txt, data/KDDTest.txt
python3 run2.py unsw  > logs/log2_unsw.txt  2>&1   # needs data/unsw_train.csv, data/unsw_test.csv
mkdir -p figs && python3 figs.py     # regenerates figs/fig*.png from the result files

#!/usr/bin/env bash
# Feature sets are explicit: "v3" = everything except the touch_ group; "v4" = everything.
set -e
cd "$(dirname "$0")"
run () { tag=$1; shift; python3 exp_lomo_train.py --tag "$tag" --drop "$@" > ../reports/train_$tag.log 2>&1;
         python3 exp_decode_grid.py --tag "$tag" --ks 2 --alphas 0.6 1.0 1.5 --out ../reports/decode_$tag.json > ../reports/decode_$tag.txt 2>&1; echo "done $tag"; }
run v4_touch __none__
run v3_nolong touch_ _m400 _m150 _past100 _fut100
run v3_nodir touch_ _past _fut
DEGRADE_DEAD_FILL=0.5 run sens_corr touch_

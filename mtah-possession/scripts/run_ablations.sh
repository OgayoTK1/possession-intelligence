#!/usr/bin/env bash
# Feature-group ablations, leave-one-match-out on the open matches. Same decoder grid for all.
set -e
cd "$(dirname "$0")"
BALL="ball_ d1_ d2_ n_ctrl n_near rel_speed dball_c nearest_is_home control_sign ctrl_density"
TEAM="cx_ cy_ sx_ sy_ speed_ vx_ xlo_ xhi_ nplay_ all_speed all_slow n_tracked"
run () { tag=$1; shift; python3 exp_lomo_train.py --tag "$tag" --drop "$@" > ../reports/train_$tag.log 2>&1;
         python3 exp_decode_grid.py --tag "$tag" --ks 2 --alphas 0.6 1.0 1.5 --out ../reports/decode_$tag.json > ../reports/decode_$tag.txt 2>&1; echo "done $tag"; }
run v3_full __none__
run v3_noball $BALL
run v3_ballonly $TEAM
run v3_nolong _m400 _m150 _past100 _fut100
run v3_nodir _past _fut

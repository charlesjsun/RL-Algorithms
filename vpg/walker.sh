#!/bin/bash
python vpg.py --env "BipedalWalker-v2" --path "models/walker_model" --save_freq 20 --epochs 200 --max_ep_len 1000 --bs 2000 --pi_lr 1e-3 --v_lr 5e-3 --v_update_steps 1 --seed 123 --hidden_layers "[100, 64]"
# python vpg.py --env "BipedalWalker-v2" --path "models/walker_model" --test_only --tests 5 --load_epoch 199
#!/bin/bash
# python vpg.py --env "CartPole-v0" --path "models/cartpole_model" --save_freq 20 --epochs 200 --max_ep_len 1000 --bs 1000 --pi_lr 1e-3 --v_lr 5e-3 --v_update_steps 1 --seed 123 --hidden_layers "[16, 16]"
python vpg.py --env "CartPole-v0" --path "models/cartpole_model" --test_only --tests 5 --load_epoch 199
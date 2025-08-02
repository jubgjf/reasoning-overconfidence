#!/usr/bin/bash

. .venv/bin/activate

accelerate launch scripts/visualize/entropy.py --layer 0
accelerate launch scripts/visualize/entropy.py --layer 5
accelerate launch scripts/visualize/entropy.py --layer 10
accelerate launch scripts/visualize/entropy.py --layer 15
accelerate launch scripts/visualize/entropy.py --layer 20
accelerate launch scripts/visualize/entropy.py --layer 25
accelerate launch scripts/visualize/entropy.py --layer 30
accelerate launch scripts/visualize/entropy.py --layer 35

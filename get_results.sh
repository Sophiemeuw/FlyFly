#!/bin/bash
set -xe

seeds=(45 38 25)
levels=(0 1 2 3 4)

rm -rf outputs/latest

for seed in "${seeds[@]}"; do
    for level in "${levels[@]}"; do
        gh run -R Sophiemeuw/FlyFly download -n simulation-outputs-level-$level-seed-$seed -D outputs/latest &
    done
done

wait
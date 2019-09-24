#!/usr/bin/env bash

for filename in gen_*.py; do
    (cd ../../ && python -m scripts.generate_rtd_images.${filename%.*})
done

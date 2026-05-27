#!/bin/bash

for size in "small" "medium"; do
    for ctx in 128 256 512; do
         echo "Profiling size: $size, ctx: $ctx..."
        nsys profile -f true -w true -t cuda,nvtx,osrt -s none \
          -o "./profiles/profile_${size}_ctx${ctx}" \
          uv run python -m benchmarks.benchmarking_e2e --size $size --context_length $ctx --mode full --num_warmup 5
    done
done
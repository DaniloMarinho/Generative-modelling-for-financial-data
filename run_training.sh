#!/bin/bash

# Define parameter values
latent_dims=(4 16 64)
g_hidden_dims=(4 16 32 64)
d_hidden_dims=(4 16 32 64)
lrs=(1e-4 5e-5)

# Iterate over combinations
for latent_dim in "${latent_dims[@]}"; do
  for g_hidden_dim in "${g_hidden_dims[@]}"; do
    for d_hidden_dim in "${d_hidden_dims[@]}"; do
      for lr in "${lrs[@]}"; do
        version="${lr}_${latent_dim}_${g_hidden_dim}_${d_hidden_dim}"

        # Run the Python command
        python3 train.py --epochs 4000 --lr $lr --latent_dim $latent_dim \
          --g_hidden_dim $g_hidden_dim --d_hidden_dim $d_hidden_dim --version "$version"
    done
  done
done

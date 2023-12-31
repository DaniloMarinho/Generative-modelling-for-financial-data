#!/bin/bash

# Define parameter values
latent_dims=(4 16 64)
g_hidden_dims=(4 16 64)
d_hidden_dims=(4 16 64)

# Iterate over combinations
for latent_dim in "${latent_dims[@]}"; do
  for g_hidden_dim in "${g_hidden_dims[@]}"; do
    for d_hidden_dim in "${d_hidden_dims[@]}"; do
      version="${latent_dim}_${g_hidden_dim}_${d_hidden_dim}"

      # Run the Python command
      python3 train_gan.py --epochs 2000 --lr 1e-4 --latent_dim $latent_dim \
        --g_hidden_dim $g_hidden_dim --d_hidden_dim $d_hidden_dim --latent_distr "normal" --version "gan $version"
    done
  done
done

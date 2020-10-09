# l2rpn_challenge
L2RPN challenge

## TO DO
 - Negative reward for making an action
 - ~~Handle MultiBinary env:~~
    - ~~Modify distribution to sample from latent_pi layer (36,) and apply mask~~
 - ~~Modify Network (CustomNetwork, see issue in github)~~
 - Add temporal dimension in obs and restriction matrix
 - Train for 1M timesteps and log into Tensorboard
 - Hyperparameter tuning
 - See how to transform from sb3 model to pure torch (for inference)
 - Log the distribution of selected actions
 - Include change_status actions

### Benchmarks

| Benchmark | Avg. episode reward | Avg. episode length |
| ----------| ------------------- | ------------------- |
| Do nothing | 585.3 (285.4) |  810.6 (408.6) |
| Random discrete | 0.434 (1.795) | 3.1 (2.7) |
| Random multibinary | 1.3 (1.8) | 4.6 (2.9) |

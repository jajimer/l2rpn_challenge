# l2rpn_challenge
L2RPN challenge

## TO DO
 - Negative reward for making an action
 - Handle MultiBinary env:
    - Some logic before agent action (mask + rules)
    - Modify distribution to sample from latent_pi layer (36,) and apply mask
 - Modify Network (CustomNetwork, see issue in github)
 - Add temporal dimension in obs and restriction matrix
 - Train for 1M timesteps and log into Tensorboard
 - Hyperparameter tuning
 - See how to transform from sb3 model to pure torch (for inference)
 - Log the distribution of selected actions

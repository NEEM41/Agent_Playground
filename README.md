# Agent_Playground
# Agent Playground

This repository contains code for exploring interactions between different agents in a grid-world environment.

![Example of agent interactions](assets/videos/multi_tiny.mp4)

## Training

You can train multiple Q-learning agents simultaneously using `train_multi_qlearning.py`. The script will save metrics, policies, and video rollouts of the evaluation environment into a `runs` directory.

To start training, run the following command:

`python train_multi_qlearning.py --map tiny_corridor --run_name threelearners --eval_every 200 --save_policy`. 

You can monitor the training progress and watch the generated videos in `runs/threelearners_tiny_corridor/videos`.

## Inference and Rendering

To render a video of different agents interacting, you can use the `render_multi_episode.py` script. This script will generate a video with various agents like `BFSAgent`, `MimicAgent`, and `QLearningAgent`.

python render_multi_episode.py --map tiny_corridor --out assets/videos/my_multi_episode.mp4This will save a video at `assets/videos/my_multi_episode.mp4`.

You can edit `render_multi_episode.py` to load your trained policies from the `runs` directory to see how they perform.

To render a video of different agents interacting, you can use the render_multi_episode.py script. This script will generate a video with various agents like BFSAgent, MimicAgent, and QLearningAgent.

`python render_multi_episode.py --map tiny_corridor --out assets/videos/my_multi_episode.mp4`

This will save a video at assets/videos/my_multi_episode.mp4.
You can edit render_multi_episode.py to load your trained policies from the runs directory to see how they perform.
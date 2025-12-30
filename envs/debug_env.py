from envs.maps import get_map
from envs.gridworld import GridWorld, GridWorldConfig

def main():
    env = GridWorld(get_map("tiny_corridor"), GridWorldConfig(max_steps=20))
    obs = env.reset(seed=0)

    print("Local view shape:", obs["local"].shape)
    print(obs["local"])

    for _ in range(5):
        a = env.action_space.sample()
        obs, r, done, info = env.step(a)
        print("step:", obs["step"])
        print(obs["local"])
        if done:
            break

if __name__ == "__main__":
    main()
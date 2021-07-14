import argparse
import pickle

import gym
from stable_baselines3 import PPO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_ckpt', required=True, type=str,
                        help='The saved ckpt to load')
    parser.add_argument('--clt', action='store_true',
                        help='Closed Loop training')
    args = parser.parse_args()

    # Load saved model
    print("Loading trained model from........")
    print(args.saved_ckpt)
    model = PPO.load(args.saved_ckpt)

    # make eval env
    print("Creating Eval env.....")
    eval_env = gym.make("L5-v0")
    eval_env.clt = args.clt

    # Check CLT
    print("CLT: ", eval_env.clt)

    # Unroll
    path = "eval_check_0_steps"  # Dummy
    obs = eval_env.reset()
    for i in range(350):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = eval_env.step(action)
        print(i, eval_env.clt)
        if done:
            break

    print(f"Saving SimulationOutput to {path}")

    with open(path + ".pkl", 'wb') as f:
        pickle.dump(info["info"], f)

import os
import shutil

import ray
import ray.rllib.agents.ppo as ppo
from l5kit.environment.envs.l5_env import L5Env
from ray.tune.registry import register_env

os.environ["L5KIT_DATA_FOLDER"] = "/home/ubuntu/level5_data"

# init directory in which to save checkpoints
chkpt_root = "tmp/exa"
shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

# init directory in which to log results
ray_results = "{}/ray_results/lr3e4".format(os.getenv("HOME"))
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

# start Ray -- add `local_mode=True` here for debugging
ray.init(ignore_reinit_error=True)

# register the custom environment
select_env = "L5-Ray-v0"
env_config_path = './gym_config.yaml'
register_env(select_env, lambda config: L5Env(env_config_path=env_config_path))

# configure the environment and create agent
config = ppo.DEFAULT_CONFIG.copy()
config["log_level"] = "WARN"
config["env_config"] = {"env_config_path": env_config_path,
                        "rescale_action": False,
                        "return_info": True}
config["framework"] = "torch"
config["num_gpus"] = 1
config["num_workers"] = 4
config["lambda"] = 0.9
config["rollout_fragment_length"] = 256
config["train_batch_size"] = 1024
config["num_sgd_iter"] = 10
config["lr"] = 3e-4
config["clip_param"] = 0.1
config["vf_clip_param"] = 10000
config["grad_clip"] = 0.5

# Trainer
trainer = ppo.PPOTrainer(config, env=select_env)

# Learn
for i in range(10000):
    # Perform one iteration of training the policy with PPO
    result = trainer.train()

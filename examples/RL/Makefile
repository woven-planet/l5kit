@ sergey_ppo:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 20000000 --save_freq 1000000 --eval_freq 500000 --n_envs 8 --n_eval_envs 16 --n_eval_episodes 1000 --kinematic -o allscenes

@ sergey_ppo_col:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 20000000 --save_freq 1000000 --eval_freq 500000 --n_envs 8 --n_eval_envs 16 --n_eval_episodes 1000 --kinematic --reward "cle" -o allscenes_col5

@ christian_ppo_alex_net:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 20000000 --save_freq 1000000 --eval_freq 500000 --n_envs 8 --n_eval_envs 16 --n_eval_episodes 1000 --kinematic --model_arch 'alex_gn' -o allscenes_alex_gn


# @ test_git_ppo:
# 	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 300000 --save_freq 100000 --eval_freq 100000 --n_envs 8 --n_eval_envs 1 --kinematic -o verify
# 	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 1000000 --save_freq 100000 --eval_freq 100000 --n_envs 8 --n_eval_envs 1 --kinematic -o verify_10scenes
# 	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 2000000 --save_freq 500000 --eval_freq 100000 --n_envs 8 --n_eval_envs 4 --n_eval_episodes 40 --kinematic -o verify_42scenes
# 	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 4000000 --save_freq 1000000 --eval_freq 200000 --n_envs 8 --n_eval_envs 4 --n_eval_episodes 100 --kinematic -o verify_100scenes
# 	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 20000000 --save_freq 1000000 --eval_freq 500000 --n_envs 8 --n_eval_envs 16 --n_eval_episodes 1000 --kinematic -o verify_allscenes

# @ test_git_col:
# 	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 4000000 --save_freq 1000000 --eval_freq 200000 --n_envs 8 --n_eval_envs 4 --n_eval_episodes 100 --kinematic --reward "cle" -o verify_100scenes_col
# 	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 4000000 --save_freq 1000000 --eval_freq 200000 --n_envs 8 --n_eval_envs 4 --n_eval_episodes 100 --kinematic --reward "cle" -o verify_100scenes_col3

# @ test_debug:
# 	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --n_steps 3000 --save_freq 500000 --eval_freq 100000 --n_envs 8 --n_eval_envs 4 --n_eval_episodes 40 --kinematic --reward "cle"
# 	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --n_steps 3000 --save_freq 500000 --eval_freq 1000 --n_envs 8 --n_eval_envs 4 --n_eval_episodes 40 --kinematic
# 	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 3000 --save_freq 500000 --eval_freq 1000 --n_envs 8 --n_eval_envs 4 --n_eval_episodes 40 --kinematic
# 	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --n_steps 3000 --save_freq 500000 --eval_freq 3000 --n_envs 8 --n_eval_envs 16 --n_eval_episodes 100 --kinematic
# 	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --n_steps 3000 --save_freq 500000 --eval_freq 50000 --n_envs 16 --n_eval_envs 4 --n_eval_episodes 40 --kinematic --model_arch 'alex_gn'

# @ test_git_sac:
# 	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python sac.py --tb_log tb_log --n_steps 100000 --save_freq 10000 --eval_freq 1000 -o verify_sac

@ sergey_ppo:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 20000000 --save_freq 1000000 --eval_freq 500000 --n_envs 8 --n_eval_envs 16 --n_eval_episodes 1000 --kinematic -o allscenes

@ sergey_ppo_col:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 20000000 --save_freq 1000000 --eval_freq 500000 --n_envs 8 --n_eval_envs 16 --n_eval_episodes 1000 --kinematic --reward "cle" -o allscenes_col5

@ christian_ppo_alex_net:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 20000000 --save_freq 1000000 --eval_freq 500000 --n_envs 8 --n_eval_envs 16 --n_eval_episodes 1000 --kinematic --model_arch 'alex_gn' -o allscenes_alex_gn

@ test_git_col:
	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 4000000 --save_freq 1000000 --eval_freq 200000 --n_envs 8 --n_eval_envs 16 --n_eval_episodes 100 --kinematic --reward "cle" -o verify_100scenes_col5_100
	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 20000000 --save_freq 1000000 --eval_freq 500000 --n_envs 8 --n_eval_envs 16 --n_eval_episodes 1000 --kinematic --reward "cle" -o verify_allscenes_pk_col5
	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 20000000 --save_freq 1000000 --eval_freq 500000 --n_envs 8 --n_eval_envs 16 --n_eval_episodes 1000 --kinematic --reward "cle" -o verify_allscenes_pk_col15
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 20000000 --save_freq 1000000 --eval_freq 500000 --n_envs 8 --n_eval_envs 16 --n_eval_episodes 1000 --kinematic --reward "cle" -o verify_allscenes_pk_col45

@ test_full_val:
	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --n_steps 2000 --save_freq 500000 --eval_freq 2000 --n_envs 8 --n_eval_envs 16 --n_eval_episodes 16000 --kinematic --enable_scene_type_aggregation --load 'logs/verify_allscenes_20000000_steps' --scene_id_to_type_path '../../dataset_metadata/validate_turns_metadata.csv'
	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --n_steps 2000 --save_freq 500000 --eval_freq 2000 --n_envs 8 --n_eval_envs 16 --n_eval_episodes 8000 --kinematic --enable_scene_type_aggregation --load 'logs/verify_allscenes_20000000_steps' --scene_id_to_type_path '../../dataset_metadata/validate_turns_metadata.csv'
	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --n_steps 2000 --save_freq 500000 --eval_freq 2000 --n_envs 8 --n_eval_envs 16 --n_eval_episodes 4000 --kinematic --enable_scene_type_aggregation --load 'logs/verify_allscenes_20000000_steps' --scene_id_to_type_path '../../dataset_metadata/validate_turns_metadata.csv'
	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --n_steps 2000 --save_freq 500000 --eval_freq 2000 --n_envs 8 --n_eval_envs 16 --n_eval_episodes 2000 --kinematic --enable_scene_type_aggregation --load 'logs/verify_allscenes_20000000_steps' --scene_id_to_type_path '../../dataset_metadata/validate_turns_metadata.csv'
	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --n_steps 2000 --save_freq 500000 --eval_freq 2000 --n_envs 8 --n_eval_envs 16 --n_eval_episodes 1000 --kinematic --enable_scene_type_aggregation --load 'logs/verify_allscenes_20000000_steps' --scene_id_to_type_path '../../dataset_metadata/validate_turns_metadata.csv'
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --n_steps 2000 --save_freq 500000 --eval_freq 2000 --n_envs 8 --n_eval_envs 16 --n_eval_episodes 4000 --kinematic --enable_scene_type_aggregation --load 'logs/verify_allscenes_20000000_steps' --scene_id_to_type_path '../../dataset_metadata/train_turns_metadata.csv'

@ overfit_s100_32: 
	# PPO_36
	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 10000000 --save_freq 500000 --eval_freq 200000 --n_envs 4 --n_eval_envs 4 --n_eval_episodes 100 --kinematic -o verify_s100_32 --eps_length 32

@ overfit_s100_16: 
	# PPO_37
	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 10000000 --save_freq 500000 --eval_freq 200000 --n_envs 4 --n_eval_envs 4 --n_eval_episodes 100 --kinematic -o verify_s100_16 --eps_length 16

@ overfit_s100_8: 
	# PPO_38
	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 10000000 --save_freq 500000 --eval_freq 200000 --n_envs 4 --n_eval_envs 4 --n_eval_episodes 100 --kinematic -o verify_s100_8 --eps_length 8
	# PPO_39
	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 1000000 --save_freq 500000 --eval_freq 100000 --n_envs 4 --n_eval_envs 1 --n_eval_episodes 1 --kinematic -o verify_8_deepcnn --eps_length 8 --model_arch 'deep_gn' 
	# PPO_40
	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 10000000 --save_freq 500000 --eval_freq 200000 --n_envs 4 --n_eval_envs 1 --n_eval_episodes 10 --kinematic -o verify_s10_lrsched_8 --eps_length 8 --lr_schedule
	# PPO_42
	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 10000000 --save_freq 400000 --eval_freq 400000 --n_envs 4 --n_eval_envs 1 --n_eval_episodes 10 --kinematic -o verify_s10_8_deepcnn --eps_length 8 --model_arch 'deep_gn' 
	# PPO_44, verify
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 6000000 --save_freq 400000 --eval_freq 400000 --n_envs 4 --n_eval_envs 1 --n_eval_episodes 10 --kinematic -o verify_sf4_yaw45_g080_deepcnn_s10_16 --eps_length 16 --gamma 0.8 --model_arch 'deep_gn2' 
	# PPO_45, verify2
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 6000000 --save_freq 400000 --eval_freq 400000 --n_envs 4 --n_eval_envs 1 --n_eval_episodes 10 --kinematic -o verify_yaw45_g080_deepcnn_s10_16 --eps_length 16 --gamma 0.8 --model_arch 'deep_gn2' 
	# PPO_49, verify
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 6000000 --save_freq 400000 --eval_freq 400000 --n_envs 4 --n_eval_envs 1 --n_eval_episodes 10 --kinematic -o verify_yaw45_s10_g080_yaw1_deepcnn_16 --eps_length 16 --gamma 0.8 --model_arch 'deep_gn2'
	# PPO_50, verify2
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 6000000 --save_freq 400000 --eval_freq 400000 --n_envs 4 --n_eval_envs 1 --n_eval_episodes 10 --kinematic -o verify_yaw45_s10_g080_yaw1_deep_16 --eps_length 16 --gamma 0.8 --model_arch 'deep_gn'

@ long_weekend:
	# PPO_51, verify
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 10000000 --save_freq 500000 --eval_freq 500000 --n_envs 4 --n_eval_envs 4 --n_eval_episodes 100 --kinematic -o verify_yaw45_s100_g080_yaw1_deepcnn_16 --eps_length 16 --gamma 0.8 --model_arch 'deep_gn2'
	# PPO_52, verify2
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 10000000 --save_freq 500000 --eval_freq 500000 --n_envs 4 --n_eval_envs 4 --n_eval_episodes 100 --kinematic -o verify_yaw45_s100_g080_yaw1_deepcnn_32 --eps_length 32 --gamma 0.8 --model_arch 'deep_gn2'
	# PPO_49, verify
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 6000000 --save_freq 400000 --eval_freq 400000 --n_envs 4 --n_eval_envs 1 --n_eval_episodes 10 --kinematic -o verify_yaw60_s10_g080_yaw1_32 --eps_length 32 --gamma 0.8
	# PPO_50, verify2
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 6000000 --save_freq 400000 --eval_freq 400000 --n_envs 4 --n_eval_envs 1 --n_eval_episodes 10 --kinematic -o verify_yaw60_s10_g080_yaw1_56x56_32 --eps_length 32 --gamma 0.8
	# PPO_52, verify2
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 6000000 --save_freq 400000 --eval_freq 400000 --n_envs 4 --n_eval_envs 1 --n_eval_episodes 10 --kinematic -o verify_yaw60_s10_g080_yaw1_32_again --eps_length 32 --gamma 0.8
	# PPO_53, verify
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 10000000 --save_freq 500000 --eval_freq 500000 --n_envs 4 --n_eval_envs 4 --n_eval_episodes 100 --kinematic -o verify_yaw60_s100_g080_yaw1_32 --eps_length 32 --gamma 0.8
	# PPO_54, verify2
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 10000000 --save_freq 500000 --eval_freq 500000 --n_envs 4 --n_eval_envs 4 --n_eval_episodes 100 --kinematic -o verify_yaw60_s100_g080_yaw1_32_cle5 --eps_length 32 --gamma 0.8 --reward 'cle'
	# PPO_55, verify
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 10000000 --save_freq 500000 --eval_freq 500000 --n_envs 4 --n_eval_envs 4 --n_eval_episodes 100 --kinematic -o verify_yaw60_s100_g080_yaw1_32_cle15 --eps_length 32 --gamma 0.8 --reward 'cle'
	# PPO_56, verify
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 10000000 --save_freq 500000 --eval_freq 500000 --n_envs 4 --n_eval_envs 4 --n_eval_episodes 100 --kinematic -o verify_yaw60_s100_g080_yaw1_32_8M_stepLR --eps_length 32 --gamma 0.8 --load 'logs/verify_yaw60_s100_g080_yaw1_32_8000000_steps' --clip_start_val 0.02 --lr 3e-5
	# PPO_57, verify2
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 10000000 --save_freq 500000 --eval_freq 500000 --n_envs 4 --n_eval_envs 4 --n_eval_episodes 100 --kinematic -o verify_yaw60_s100_g080_yaw1_32_run2 --seed 10 --eps_length 32 --gamma 0.8
	# PPO_58, verify
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 10000000 --save_freq 500000 --eval_freq 500000 --n_envs 4 --n_eval_envs 4 --n_eval_episodes 100 --kinematic -o verify_yaw60_s100_g080_yaw1_32_run3 --seed 20 --eps_length 32 --gamma 0.8
	# PPO_59, verify2
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --tb_log tb_log --n_steps 10000000 --save_freq 500000 --eval_freq 500000 --n_envs 4 --n_eval_envs 4 --n_eval_episodes 100 --kinematic -o verify_yaw60_s100_g080_yaw1_32_run2_stepLR_75M --seed 10 --eps_length 32 --gamma 0.8 --load 'logs/verify_yaw60_s100_g080_yaw1_32_run2_7500000_steps' --clip_start_val 0.025 --lr 3e-5
	# PPO_60, verify
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo_training.py --tb_log tb_log --n_steps 10000000 --save_freq 500000 --eval_freq 500000 --n_envs 4 --n_eval_envs 8 --n_eval_episodes 1000 --kinematic -o verify_yaw45_s100_g080_yaw1_32_sample --eps_length 32 --gamma 0.8 --training_scheme 'weighted_sampling' --enable_scene_type_aggregation --scene_id_to_type_path '../../dataset_metadata/train_turns_metadata.csv'

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
	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --n_steps 3000 --save_freq 500000 --eval_freq 50000 --n_envs 16 --n_eval_envs 4 --n_eval_episodes 40 --kinematic --model_arch 'alex_gn'
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo.py --n_steps 10000 --save_freq 500000 --eval_freq 50000 --n_envs 4 --n_eval_envs 4 --n_eval_episodes 40 --kinematic
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo_training.py --n_steps 10000 --save_freq 500000 --eval_freq 50000 --n_envs 4 --n_eval_envs 4 --n_eval_episodes 40 --kinematic

# @ test_git_sac:
# 	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python sac.py --tb_log tb_log --n_steps 100000 --save_freq 10000 --eval_freq 1000 -o verify_sac


	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo_training.py --n_steps 2000 --save_freq 500000 --eval_freq 2000 --n_envs 4 --n_eval_envs 4 --n_eval_episodes 100 --kinematic --load 'logs/verify_yaw45_s100_g080_yaw1_32_run3_75M_stepLR_3000000_steps' --gamma 0.8 --output DEL


	# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo_training.py --n_steps 2000 --save_freq 500000 --eval_freq 2000 --n_envs 4 --n_eval_envs 4 --n_eval_episodes 1000 --kinematic --load 'logs/verify_yaw45_s100_g080_yaw1_32_run3_75M_stepLR_3000000_steps' --gamma 0.8 --output DEL --enable_scene_type_aggregation --scene_id_to_type_path '../../dataset_metadata/validate_turns_metadata.csv' --training_scheme 'weighted_sampling'
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo_training.py --n_steps 2000 --save_freq 500000 --eval_freq 2000 --n_envs 4 --n_eval_envs 4 --n_eval_episodes 1000 --kinematic --load 'logs/verify_yaw45_s100_g080_yaw1_32_run3_75M_stepLR_3000000_steps' --gamma 0.8 --output DEL --training_scheme 'weighted_reward'

@simnet:
	# PPO 66
 	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python ppo_training.py --tb_log tb_log --n_steps 10000000 --save_freq 500000 --eval_freq 500000 --n_envs 4 --n_eval_envs 4 --n_eval_episodes 100 --kinematic -o verify_yaw45_s100_g080_yaw1_32_simnet --eps_length 32 --gamma 0.8 --simnet --simnet_model_path "/code/parth/dataset/dn_h3_p05_1999999_steps.pt"

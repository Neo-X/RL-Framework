
# You can use a name instead of the ID if you named your account previously
# Get organization name
export ORG_NAME=$(eai organization  get --fields name --no-header)
# Get account name
export ACCOUNT_NAME=$(eai account get --fields name --no-header)
export ACCOUNT_ID=$ORG_NAME.$ACCOUNT_NAME

# eai data push $ORG_NAME.$ACCOUNT_NAME.rlframe /home/gberseth/playground/RL-Framework/
# eai data push $ORG_NAME.$ACCOUNT_NAME.doodad /home/gberseth/playground/doodad_vitchry/
# eai data push $ORG_NAME.$ACCOUNT_NAME.motionimitation /home/gberseth/playground/motion_imitation/
# eai data push $ORG_NAME.$ACCOUNT_NAME.stanquad /home/gberseth/playground/StanfordQuadruped/
# eai data push $ORG_NAME.$ACCOUNT_NAME.learning_data /home/gberseth/learning_data_eai/

for i in `seq 1 2`;
    do
	eai job submit --preemptable\
	    --data $ORG_NAME.$ACCOUNT_NAME.rlframe:/home/gberseth/playground/RL-Framework/ \
	    --data $ORG_NAME.$ACCOUNT_NAME.doodad:/home/gberseth/playground/doodad/ \
	    --data $ORG_NAME.$ACCOUNT_NAME.motionimitation:/home/gberseth/playground/motion_imitation/ \
	    --data $ORG_NAME.$ACCOUNT_NAME.stanquad:/home/gberseth/playground/StanfordQuadruped/ \
	    --data $ORG_NAME.$ACCOUNT_NAME.learning_data:/home/gberseth/learning_data/ \
	    --image gberseth/rlframe:latest \
	    -e PYTHONPATH=/home/gberseth/playground/RL-Framework/:/opt/TerrainRLSim:/home/gberseth/playground/doodad/:/home/gberseth/playground/motion_imitation/motion_imitation/:/home/gberseth/playground/StanfordQuadruped/ \
	    --gpu 1 --cpu 5 --mem 16 \
	    -- bash -c "pushd /home/gberseth/playground/RL-Framework/; python3 trainModel.py --config=settings/terrainRLImitate/PPO/Imitation_Learning_GRF_StanQuad_1Sub_LSTM_FD_Reward_Dual_Encode_Decode_VAE_2State_2_Advisarial_BCE_refresh.json -p 5 --shouldRender=false --log_comet=true --on_policy=fast --print_level=train --random_seed=$i"
	    # -- bash -c "pushd /home/gberseth/playground/RL-Framework/; python3 trainModel.py --config=settings/terrainRLImitate/PPO/Imitation_Learning_GRF_UniTree_1Sub_LSTM_FD_Reward_Dual_Encode_Decode_VAE_2State_2_Advisarial_BCE_refresh2.json -p 5 --shouldRender=false --log_comet=true --on_policy=fast --print_level=train --random_seed=$i"
	    # -- bash -c "pushd /home/gberseth/playground/RL-Framework/; python3 trainModel.py --config=settings/terrainRLImitate/TD3/Imitation_Learning_GRF_Humanoid2D.json -p 5 --shouldRender=false --log_comet=true --on_policy=fast --print_level=hyper_train --random_seed=$i"
	    # -- bash -c 'pushd /home/gberseth/playground/RL-Framework/; python3 trainModel.py --config=settings/terrainRLImitate/PPO/Imitation_Learning_GRF_StanQuad.json -p 5 --shouldRender=false --log_comet=true --on_policy=fast --print_level=train'
	    # --gpu 1 --cpu 5 --mem 16 --name test_eai_terrainrl_laikago_ppo1 \
	    # -- bash -c 'pushd /home/gberseth/playground/RL-Framework/; python3 trainModel.py --config=settings/terrainRLImitate/PPO/Imitation_Learning_GRF_UniTree.json -p 5 --shouldRender=false --shouldRender=false --log_comet=true --on_policy=fast --print_level=train'
	    # -- bash -c 'pushd /home/gberseth/playground/RL-Framework/; python3 trainModel.py --config=settings/terrainRLImitate/TD3/Imitation_Learning_GRF_UniTree.json -p 6 --shouldRender=false --shouldRender=false --log_comet=true --on_policy=fast --print_level=train'
	    # -- bash -c 'pushd /home/gberseth/playground/RL-Framework/; python3 trainModel.py --config=settings/terrainRLImitate/PPO/settings/terrainRLImitate/PPO/Imitation_Learning_GRF_UniTree_1Sub_LSTM_FD_Reward_Dual_Encode_Decode_VAE_2State_2_Advisarial_BCE_refresh2.json -p 5 --shouldRender=false --log_comet=true --on_policy=fast --print_level=train'    
        echo $i
    done 

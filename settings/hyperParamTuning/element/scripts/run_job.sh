
# You can use a name instead of the ID if you named your account previously
# Get organization name
export ORG_NAME=$(eai organization  get --fields name --no-header)
# Get account name
export ACCOUNT_NAME=$(eai account get --fields name --no-header)
export ACCOUNT_ID=$ORG_NAME.$ACCOUNT_NAME

eai data push $ORG_NAME.$ACCOUNT_NAME.rlframe /home/gberseth/playground/RL-Framework/
eai data push $ORG_NAME.$ACCOUNT_NAME.doodad /home/gberseth/playground/doodad_vi/

eai job submit --preemptable\
    --data $ORG_NAME.$ACCOUNT_NAME.rlframe:/home/gberseth/playground/RL-Framework/ \
    --data $ORG_NAME.$ACCOUNT_NAME.doodad:/home/gberseth/playground/doodad_vitchry/ \
    --image gberseth/rlframe:latest \
    -e PYTHONPATH=/home/gberseth/playground/RL-Framework/:/home/gberseth/playground/doodad/ \
    --gpu 1 --cpu 8 --mem 16 --name test_eai_terrainrl6 \
    -- bash -c 'pushd /home/gberseth/playground/RL-Framework/; python3 trainModel.py --config=settings/terrainRLImitate/PPO/Imitation_Learning_GRF_UniTree.json -p 6 --shouldRender=false --shouldRender=false --log_comet=true --on_policy=fast --print_level=train'

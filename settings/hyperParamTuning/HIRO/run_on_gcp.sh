#!/bin/bash
#gcloud compute instances create btrabucco-rl-framework \
#     --image rl-framework-image \
#     --image-project glen-rl-framework \
#     --machine-type=n1-standard-4 \
#     --boot-disk-size=50GB \
#     --preemptible
cmd='docker run --rm -it us.gcr.io/glen-rl-framework/glen:latest2 bash -c "pushd /opt/RL-Framework; ./update_and_compile.sh; python3 tuneHyperParameters.py --config=settings/navgame2D/TD3/Two_Level_HIRO_TD3.json -p 2 --metaConfig=settings/hyperParamTuning/HIRO/sweep_num_goals_to_resample.json --plot=false --shouldRender=false --meta_sim_samples 3 --meta_sim_threads 6"'
gcloud compute ssh btrabucco-rl-framework -- $cmd
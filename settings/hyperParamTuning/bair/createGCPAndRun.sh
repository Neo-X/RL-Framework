#!/bin/bash 


### Create instance
# gcloud compute instances create rl-frame-temp-inst-0 \
#     --image rl-framework-image \
#     --image-project glen-rl-framework \
#     --machine-type=n1-standard-4 \
#     --boot-disk-size=50GB \
#     --preemptible
    
### Run command on instance
## gcloud compute ssh rl-frame-temp-inst-0 --command 'docker run --rm us.gcr.io/glen-rl-framework/glen:latest2 bash -c "pushd /opt/RL-Framework; python3 trainModel.py --config=tests/settings/gapGame2D/PPO/FixedSTD_Tensorflow-v2.json -p 2 --metaConfig=settings/hyperParamTuning/element/exploration_rate.json --plot=false --shouldRender=false --bootstrap_sample=1 " '
cmd='docker run --rm -it us.gcr.io/glen-rl-framework/glen:latest2 bash -c "pushd /opt/RL-Framework; ./update_and_compile.sh; python3 trainModel.py --config=tests/settings/gapGame2D/PPO/FixedSTD_Tensorflow-v2.json -p 2 --metaConfig=settings/hyperParamTuning/element/exploration_rate.json --plot=false --shouldRender=false --bootstrap_sample=1"'
gcloud compute ssh rl-frame-temp-inst-0 -- $cmd
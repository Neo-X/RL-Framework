from doodad.launch import launch_api
from doodad import mode
from doodad import mount
import sys

cmd = sys.argv[1]
print ("cmd: ", cmd)

"""
docker run --rm -it us.gcr.io/glen-rl-framework/glen:latest2 bash -c 'pushd /opt/RL-Framework; ./update_and_compile.sh; python3 trainModel.py --config=tests/settings/gapGame2D/PPO/FixedSTD_Tensorflow-v2.json -p 2 --metaConfig=settings/hyperParamTuning/element/exploration_rate.json --plot=false --shouldRender=false --bootstrap_sample=1 --experiment_logging="{\"use_comet\": true, \"project_name\": \"vizimitation\"}"'

python3 runDoodad.py 'pushd /opt/RL-Framework; ./update_and_compile.sh; python3 trainModel.py --config=tests/settings/gapGame2D/PPO/FixedSTD_Tensorflow-v2.json -p 2 --metaConfig=settings/hyperParamTuning/element/exploration_rate.json --plot=false --shouldRender=false --bootstrap_sample=1 --experiment_logging="{\"use_comet\": true, \"project_name\": \"vizimitation\"}"'

gcloud compute instances create [INSTANCE_NAME] \
    --image rl-framework-image \
    --image-project glen-rl-framework \
    --machine-type=n1-standard-4 \
    --boot-disk-size=50GB \
    --preemptible


"""

useGCP = True
if (useGCP):
    mode = mode.GCPMode(
            gcp_bucket='rl-framework-cluster-bucket',
            gcp_log_path='/doodad/logs/',
            gcp_project='glen-rl-framework',
            gcp_image='rl-framework-image',
            gcp_image_project='glen-rl-framework',
            instance_type='n1-standard-4', ## 
            zone='us-west1-a'
        )
    mnt = mount.MountGCP(
        gcp_path='./doodad',
        mount_point='/opt/doodad',  # Directory visible to the running job.
        output=True
    )
else:
    mode = mode.LocalMode()
    mnt = mount.MountLocal(
        local_dir='/home/gberseth/playground/RL-Framework/doodad',
        mount_point='/opt/doodad',
        output=True
        )

out = launch_api.run_command(
    command=cmd,
    docker_image='us.gcr.io/glen-rl-framework/glen:latest2',
    mode=mode,
    mounts=[mnt]
)

print (out)
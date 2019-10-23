from doodad.launch import launch_api
from doodad import mode
from doodad import mount
import sys

cmd = sys.argv[1]
print("cmd: ", cmd)

"""

python3 runDoodad.py 'pushd /opt/RL-Framework; ./update_and_compile.sh; python3 tuneHyperParameters.py --config settings/navgame2D/TD3/Two_Level_HIRO_TD3.json --metaConfig=settings/hyperParamTuning/HIRO/sweep_num_goals_to_resample.json -p 4 --meta_sim_samples 3 --meta_sim_threads 1  --plot=false --shouldRender=false'

"""

useGCP = True
if (useGCP):
    mode = mode.GCPMode(
            gcp_bucket='rl-framework-cluster-bucket',
            gcp_log_path='/doodad/logs2/',
            gcp_project='glen-rl-framework',
            gcp_image='rl-framework-image',
            gcp_image_project='glen-rl-framework',
            instance_type='n1-standard-4',
            terminate_on_end=True,
            preemptible=True,
            zone='us-west1-a'
        )
    mnt = mount.MountGCP(
        gcp_path='./doodad',
        mount_point='/opt/doodad',  # Directory visible to the running job.
        output=True)

else:
    mode = mode.LocalMode()
    mnt = mount.MountLocal(
        local_dir='/home/btrabucco/research/doodad',
        mount_point='/opt/doodad',
        output=True)

out = launch_api.run_command(
    command=cmd,
    docker_image='us.gcr.io/glen-rl-framework/glen:latest2',
    mode=mode,
    mounts=[mnt]
)

print (out)

from doodad.launch import launch_api
from doodad import mode
import sys

cmd = sys.argv[1]
print ("cmd: ", cmd)

useGCP = True
if (useGCP):
    mode = mode.GCPMode(
            gcp_bucket='rl-framework-cluster-bucket',
            gcp_log_path='/doodad/logs/',
            gcp_project='glen-rl-framework',
            gcp_image='rl-framework-image',
            gcp_image_project='glen-rl-framework',
            instance_type='n1-standard-2', ## 
            zone='us-west1-a'
        )
else:
    mode = mode.LocalMode()

launch_api.run_command(
    command=cmd,
    docker_image='us.gcr.io/glen-rl-framework/glen:latest2',
    mode=mode
)
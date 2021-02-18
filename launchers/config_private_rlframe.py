
BASE_DIR = '/home/gberseth/'
CODE_DIRS_TO_MOUNT = [
    BASE_DIR + 'playground/RL-Framework',
    BASE_DIR + 'playground/RLSimulationEnvironments',
#     BASE_DIR + 'playground/motion_imitation/motion_imitation',
#     '/home/gberseth/playground/StanfordQuadruped',
]
NON_CODE_DIRS_TO_MOUNT = [
]
# LOCAL_LOG_DIR = '/tmp/doodad-output/'
LOCAL_LOG_DIR = BASE_DIR + 'learning_data/rlframe/'
OUTPUT_DIR_FOR_DOODAD_TARGET = '/tmp/doodad-output/'
DIR_AND_MOUNT_POINT_MAPPINGS = [
    dict(
        local_dir=BASE_DIR + '/.mujoco/',
        mount_point='/root/.mujoco',
    ),
    dict(
        local_dir=BASE_DIR + '/playground/TerrainRLSim/args/envs.json',
        mount_point='/opt/TerrainRLSim/args/envs.json',
    ),
#     dict(
#         local_dir='/home/gberseth/playground/CoMPS',
#         mount_point='/root/playground/CoMPS',
#     ),
#     dict(
#         local_dir='/home/gberseth/playground/doodad_vitchry',
#         mount_point='/root/playground/doodad',
#     ),
]

"""
AWS Settings
"""
AWS_S3_PATH = 's3://comps-test/rlframe'

# The docker image is looked up on dockerhub.com.
DOODAD_DOCKER_IMAGE = 'gberseth/rlframe:latest'
INSTANCE_TYPE = 'c5.2xlarge' # 8 core 16 gb mem
SPOT_PRICE = 0.13

GPU_DOODAD_DOCKER_IMAGE = 'TODO'
GPU_INSTANCE_TYPE = 'g3.4xlarge'
GPU_SPOT_PRICE = 0.5
REGION_TO_GPU_AWS_IMAGE_ID = {
    'us-east-2': "ami-024f0e84c5a8282a8",
#     'us-east-1': "ami-ce73adb1",
}
AWS_FILE_TYPES_TO_SAVE = (
    '*.txt', '*.csv', '*.json', '*.gz', '*.tar',
    '*.log', '*.pkl', '*.mp4', '*.png', '*.jpg',
    '*.jpeg', '*.patch', '*.html', "*.h5", "*.svg", "*.mp4"
)

"""
SSH Settings
"""
SSH_HOSTS = dict(
    default=dict(
        username='gberseth',
        hostname='localhost',
#         hostname='192.168.111.123',
    ),
    crete=dict(
        username='gberseth',
        hostname='crete',
#         hostname='192.168.111.123',
    ),
    paros=dict(
        username='gberseth',
        hostname='paros',
#         hostname='192.168.111.123',
    ),
)
SSH_DEFAULT_HOST = 'gberseth'
SSH_PRIVATE_KEY = '~/.ssh/id_rsa'
SSH_LOG_DIR = '~/_shared/res'
SSH_TMP_DIR = '~/_shared/tmp'

"""
Local Singularity Settings
"""
SINGULARITY_IMAGE = 'TODO'
SINGULARITY_PRE_CMDS = [
]


"""
BRC/Slurm Settings

These are basically the same settings as above, but for the remote machine
where you will be running the generated script.
"""
SLURM_CONFIGS = dict(
    cpu=dict(
        account_name='TODO',
        partition='TODO',
        n_gpus=0,
        max_num_cores_per_node=8,
    ),
    gpu=dict(
        account_name='TODO',
        partition='TODO',
        n_gpus=1,
        max_num_cores_per_node=8,
        n_cpus_per_task=2,
    ),
)
BRC_EXTRA_SINGULARITY_ARGS = '--writable -B /usr/lib64 -B /var/lib/dcv-gl'
TASKFILE_PATH_ON_BRC = 'TODO'


SSS_CODE_DIRS_TO_MOUNT = [
]
SSS_NON_CODE_DIRS_TO_MOUNT = [
]
SSS_LOG_DIR = '/global/scratch/vitchyr/doodad-log'


SSS_GPU_IMAGE = 'TODO'
SSS_CPU_IMAGE = 'TODO'
SSS_RUN_DOODAD_EXPERIMENT_SCRIPT_PATH = 'TODO'
SSS_PRE_CMDS = [
    'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH'
]


"""
GCP Settings

To see what zones support GPU, go to https://cloud.google.com/compute/docs/gpus/
"""
GCP_IMAGE_NAME = 'TODO'
GCP_GPU_IMAGE_NAME = 'TODO'
GCP_BUCKET_NAME = 'TODO'

GCP_DEFAULT_KWARGS = dict(
    zone='us-west1-a',
    instance_type='n1-standard-4',
    image_project='TODO',
    terminate=True,
    preemptible=True,  # is much more expensive!
    gpu_kwargs=dict(
        gpu_model='nvidia-tesla-k80',
        num_gpu=1,
    )
)
GCP_FILE_TYPES_TO_SAVE = (
    '*.txt', '*.csv', '*.json', '*.gz', '*.tar',
    '*.log', '*.pkl', '*.mp4', '*.png', '*.jpg',
    '*.jpeg', '*.patch', '*.html'
)

# Overwrite with private configurations
try:
    from doodad.easy_launch.config_private import *
except ImportError as e:
    from doodad.utils import REPO_DIR
    import os.path as osp
    command_to_run = "cp {} {}".format(
        __file__,
        __file__[:-3] + '_private.py',
        )
    print("You should set up the private config files. Run:\n\n  {}\n".format(
        command_to_run
    ))

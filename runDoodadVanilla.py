from doodad.launch import launch_api
from doodad import mode
from doodad import mount
import os


if __name__ == "__main__":

    local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doodad_mount")
    remote_code_dir = "remote_mount"
    target_py_file = os.path.join(remote_code_dir, "sleep_five_minutes.py")
    doodad_command = "python {}".format(target_py_file)

    print(dict(
        local_dir=local_dir,
        remote_code_dir=remote_code_dir,
        target_py_file=target_py_file,
        doodad_command=doodad_command))

    local_mode = mode.LocalMode()
    gcp_mode = mode.GCPMode(
        gcp_project='glen-rl-framework',
        gcp_bucket='rl-framework-cluster-bucket',
        gcp_log_path='gcp_test',
        gcp_image='ubuntu-1604-xenial-v20191010',
        gcp_image_project='ubuntu-os-cloud',
        instance_type='n1-standard-4',
        terminate_on_end=True,
        zone='us-west1-a')

    # replicate the folder structure of the local directory
    local_mount = mount.MountLocal(
        local_dir,
        mount_point=remote_code_dir,
        output=False)

    print(launch_api.run_command(
        command=doodad_command,
        mode=gcp_mode,
        mounts=(local_mount,)))

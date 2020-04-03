# docker build --default-runtime=nvidia -f Dockerfile_trl -t rlframe_trl:latest .
# docker run --rm --runtime=nvidia -v /opt/OpenGL/:/usr/lib/nvidia -it rlframe_trl:latest /bin/bash -c "pushd /root/playground/RL-Framework; git pull origin master; python3 trainModel.py --config=settings/MiniGrid/TagEnv/PPO/Tag_SLAC_mini.json -p 4 --bootstrap_samples=2000 --max_epoch_length=16 --rollouts=4 --skip_rollouts=true --train_actor=false --train_critic=false --epochs=32 --fd_updates_per_actor_update=64 --on_policy=fast"

### Not using nvidia/cuda for now
docker build -f Dockerfile_smirl -t rlframe_smirl:latest .

docker run --rm -it rlframe_smirl:latest /bin/bash -c "pushd /root/playground/RLSimulationEnvironments/; git pull origin master; popd; pushd /root/playground/RL-Framework; git pull origin master; python3 trainModel.py --config=settings/MiniGrid/TagEnv/PPO/Tag_SLAC_mini.json -p 4 --bootstrap_samples=2000 --max_epoch_length=16 --rollouts=4 --skip_rollouts=true --train_actor=false --train_critic=false --epochs=32 --fd_updates_per_actor_update=64 --on_policy=fast"

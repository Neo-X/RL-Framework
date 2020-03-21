docker build -f Dockerfile_trl -t rlframe_trl:latest .

docker run --rm -v /opt/OpenGL/:/usr/lib/nvidia -it rlframe_trl:latest /bin/bash -c "pushd /root/playground/RL-Framework; git pull origin master; trainModel.py --config=settings/terrainRLImitate/PPO/SLAC.json -p 4 --bootstrap_samples=2000 --max_epoch_length=32 --rollouts=32 --skip_rollouts=false --on_policy=fast"

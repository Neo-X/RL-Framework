docker build -f Dockerfile -t rlframe:latest .

docker run --rm -it rlframe:latest /bin/bash -c "pushd /root/playground/RL-Framework; git pull origin master; ls; python3 trainModel.py --config=settings/miniGrid/simpleRoomLatent/DQN/Tensorflow_SimpleRoom_LatentState.json"

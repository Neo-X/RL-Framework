docker build -f Dockerfile_smirl -t rlframe_smirl:latest .

docker run --rm -it rlframe_smirl:latest /bin/bash -c "pushd /root/playground/BayesianSurpriseCode; git pull origin master; popd; pushd /root/playground/RL-Framework; git pull origin master; python3 trainModel.py --config=settings/MiniGrid/simpleRoomLatent/DQN/Tensorflow_SimpleRoom_HMM_Marginal.json -p 2 --shouldRender=false"

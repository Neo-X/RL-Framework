
aws s3 sync --exclude="*.mp4" --exclude="params.pkl" --exclude="*.log" s3://comps-test/rlframe/ ~/learning_data/rlframe/
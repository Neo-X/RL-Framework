
# Intro

This folder contains some scripts and files to run simulations in a docker container.

  
## Getting Started

### Using Ray

starting the cluster

ray up settings/hyperParamTuning/bair/RL-Framework_Cluster.yaml

## Docker

### Push docker update

docker commit d0fb7d3cea01 glen:latest; docker tag glen:latest us.gcr.io/glen-rl-framework/glen:latest; docker push us.gcr.io/glen-rl-framework/glen:latest


### SSH into head node to install kubernetes
gcloud compute ssh cluster-head

sudo apt-get install kubectl

### Get credentials to access the cluster from the head node
gcloud container clusters get-credentials private-cluster --zone us-west1-c


### Verify the nodes can be accessed
kubectl get nodes --output yaml | grep -A4 addresses
or
kubectl get nodes --output wide

### Delete the cluster
gcloud container clusters delete private-cluster --zone us-west1-c
### Delete head node
gcloud container instances delete cluster-head --zone us-west1-c

### Pushing docker containers to registry

https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app


### Kubernetes Commands

Run a job
```
kubectl apply -f test.yaml
```
Where test.yaml is a manifest file that dictates most of the job details.

Describe the state of a current joc
```
kubectl describe job example-job2
```

List the current pods
```
kubectl get pods -a
```

Get the log from a "pod"
```
kubectl logs example-job2-45cs9
```

### Mounting shared storage in VM
```
mkdir /Cluster_Bucket
sudo chmod a+w /Cluster_Bucket/
```
Need to get /etc/fuse.conf, uncomment user_allow_other
```
gcsfuse --dir-mode "777" -o allow_other rl-framework-cluster-bucket /Cluster_Bucket/
```

```
export GOOGLE_APPLICATION_CREDENTIALS="/home/gberseth/Glen-RL-Framework-19b1aadaca75.json" gcsfuse --dir-mode "777" -o allow_other rl-framework-cluster-bucket /mnt/data
```
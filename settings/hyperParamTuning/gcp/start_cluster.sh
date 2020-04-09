#!/bin/bash
### Creates a private cluster on gcp using the normal head-node

gcloud compute instances start cluster-head --zone us-west1-c

### Create the private cluster
gcloud beta container clusters create private-cluster \
    --enable-private-nodes \
    --master-ipv4-cidr 172.16.0.16/28 \
    --enable-ip-alias \
    --preemptible \
    --enable-private-nodes \
    --num-nodes 2 --machine-type n1-standard-2 \
    --enable-autoscaling --min-nodes=0 --max-nodes=3 \
    --zone us-west1-c \
    --create-subnetwork ""
    

masterNodeIP=$(gcloud compute instances describe cluster-head | grep natIP | awk '{print $2}')
echo $masterNodeIP

### Add head node access
gcloud container clusters update private-cluster \
    --enable-master-authorized-networks \
    --master-authorized-networks $masterNodeIP/32
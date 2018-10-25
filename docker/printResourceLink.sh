#!/bin/bash
###
### Print list of resouce links so you can quickly look through them
###

### Get active jobs
activeJobs="$(borgy ps --state alive | cut -d' ' -f1 | tail -n +2)"

### For each active job print just the resource link
for jobb in $activeJobs;
do 
	borgy info $jobb | grep resourceUsageUrl
done

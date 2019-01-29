#!/bin/bash

### Delete all the exp memory buffer that are stored.
### double check what is being deleted
find . -name "*expBufferInit.hdf5" -type f
find . -name "agent_FD_expBufferInit.hdf5" -type f
### Actually delete everything
# find . -name "*expBufferInit.hdf5" -type f -delete
# find . -name "agent_FD_expBufferInit.hdf5" -type f -delete

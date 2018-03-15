## Intro

These config files are used to setup the desire type of simulation. 
There is a large number of possible settings and new ones can be added easily.
A copy of these settings are passed throughout the code so you can easily add more simulation configuration in the file that can be used in the code.


### Info on some of the settings

num_available_threads

| Param name  | description | data type |
|-----------------|-------------|----|
| num_available_threads         |  This parameter controls the number of thread that can be used to run simulations in parallel. This value can be set to -1 in order to avoid using some of the threading code for simulation, making many parts synchronous  |  int |
| model_type         |  A string to determine the type of network model to use for the critic and policy net. This string currently can be used to directly specifiy the classname and path to the network model to load.  |  string  |



## Intro

These config files are used to setup the desire type of simulation. 
There is a large number of possible settings and new ones can be added easily.
A copy of these settings are passed throughout the code so you can easily add more simulation configuration in the file that can be used in the code.


### Info on some of the settings

num_available_threads

| Param name  | description | data type |
|-----------------|-------------|----|
| biped3d_step         |    A controller where the actions and number of actions correspond to the number of links in the simulated controlled character  |  yes |
| biped3d_sym_step         |     A controller where the actions and number of actions correspond to the number of links in the simulated controlled character. This controller assumes a symmetry in the controller and works by only learning part of the state space that would correspond to only ever taking right steps. When a left step is taking the state is mirrored across the sagittal plane.  |  yes  |
|                 |                |    | 


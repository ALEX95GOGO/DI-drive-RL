#!/bin/bash
# Open a new gnome-terminal and run the first command
#gnome-terminal -- bash -c "singularity exec --nv /data/zhuzhuan/Download/carla-0.9.10.1.sif /home/carla/CarlaUE4.sh -opengl --carla-world-port=9000; exec bash" 
#gnome-terminal -- bash -c "singularity exec --nv /data/zhuzhuan/Download/carla-0.9.10.1.sif /home/carla/CarlaUE4.sh -opengl --carla-world-port=9002; exec bash" 
#gnome-terminal -- bash -c "singularity exec --nv /data/zhuzhuan/Download/carla-0.9.10.1.sif /home/carla/CarlaUE4.sh -opengl --carla-world-port=9004; exec bash" 
#gnome-terminal -- bash -c "singularity exec --nv /data/zhuzhuan/Download/carla-0.9.10.1.sif /home/carla/CarlaUE4.sh -opengl --carla-world-port=9006; exec bash" 
#gnome-terminal -- bash -c "singularity exec --nv /data/zhuzhuan/Download/carla-0.9.10.1.sif /home/carla/CarlaUE4.sh -opengl --carla-world-port=9008; exec bash" 


gnome-terminal -- bash -c "SINGULARITYENV_SDL_VIDEODRIVER=offscreen singularity exec --nv -e ../../../carla-0.9.11.sif /home/carla/CarlaUE4.sh -carla-port=2000; exec bash" 
#gnome-terminal -- bash -c "SINGULARITYENV_SDL_VIDEODRIVER=offscreen singularity exec --nv -e ../../../carla-0.9.11.sif /home/carla/CarlaUE4.sh -carla-port=9002; exec bash" 
#gnome-terminal -- bash -c "SINGULARITYENV_SDL_VIDEODRIVER=offscreen singularity exec --nv -e ../../../carla-0.9.11.sif /home/carla/CarlaUE4.sh -carla-port=9004; exec bash" 
#gnome-terminal -- bash -c "SINGULARITYENV_SDL_VIDEODRIVER=offscreen singularity exec --nv -e ../../../carla-0.9.11.sif /home/carla/CarlaUE4.sh -carla-port=9006; exec bash" 
#gnome-terminal -- bash -c "SINGULARITYENV_SDL_VIDEODRIVER=offscreen singularity exec --nv -e ../../../carla-0.9.11.sif /home/carla/CarlaUE4.sh -carla-port=9008; exec bash" 

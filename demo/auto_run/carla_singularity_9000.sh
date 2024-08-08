#!/bin/bash
# Open a new gnome-terminal and run the first command
gnome-terminal -- bash -c "singularity exec --nv /data/zhuzhuan/Download/carla-0.9.13.sif /home/carla/CarlaUE4.sh --carla-world-port=9000; exec bash" 
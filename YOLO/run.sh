 #!/bin/bash

#SBATCH --mem 50GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -p gpunode
#SBATCH -G 1


export MPLCONFIGDIR=/scratch/mpavlov-YOLO/tmp
export YOLO_CONFIG_DIR=/scratch/mpavlov-YOLO/tmp
yolo segment train data=config.yaml model=yolov8n-seg.yaml epochs=400 imgsz=800 cache=True batch=16 optimizer=RAdam

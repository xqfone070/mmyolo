#!/bin/sh
# empty nohup.out
echo "" > nohup.out
export CUDA_VISIBLE_DEVICES=0,1,2,3
comma_num=`echo $CUDA_VISIBLE_DEVICES | grep -o "," | wc -l`
gpu_num=`expr $comma_num + 1`


if [ $gpu_num -eq 1 ]
then
  dist_train=false
else
  dist_train=true
fi



config_file='configs/alex_thz_item_det/yolov8_l.py'
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "gpu_num=$gpu_num"
echo "dist_train=$dist_train"
echo "config_file=$config_file"

if [ "$dist_train" = true ]
then
  chmod +x tools/dist_train.sh
  dos2unix tools/dist_train.sh
  nohup tools/dist_train.sh $config_file $gpu_num &
else
  nohup python tools/train.py $config_file &
fi

# tailf nohup.out
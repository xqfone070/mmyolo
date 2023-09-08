#!/bin/sh
# empty nohup.out
# echo "" > nohup.out
#log file
# shellcheck disable=SC2006
date_str=`date +"%Y%m%d_%H%M%S"`
log_dir="logs_nohup"
mkdir -p $log_dir
log_file=${log_dir}/nohup_${date_str}.log
log_link_file='nohup.log'

# gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3
comma_num=`echo $CUDA_VISIBLE_DEVICES | grep -o "," | wc -l`
gpu_num=`expr $comma_num + 1`


if [ $gpu_num -eq 1 ]
then
  dist_train=false
else
  dist_train=true
fi


# config_file='configs_alex/thz_body_det/yolov5.py'
# config_file='configs_alex/thz_item_det/yolov8.py'
config_file='configs_alex/thz_body_det/yolov8.py'
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "gpu_num=$gpu_num"
echo "dist_train=$dist_train"
echo "config_file=$config_file"

if [ "$dist_train" = true ]
then
  chmod +x tools/dist_train.sh
  dos2unix tools/dist_train.sh
  nohup tools/dist_train.sh $config_file "$gpu_num" >"$log_file" 2>&1 &
else
  nohup python tools/train.py $config_file >"$log_file" 2>&1 &
fi

sleep 1
ln -sf "$log_file" $log_link_file
# tail -f "$log_file"



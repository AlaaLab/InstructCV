#!/bin/bash

# 定义保存内存占用率的文件路径
log_file="/lustre/grp/gyqlab/lism/brt/language-vision-interface/memory.log"

# 循环打印内存占用率并保存到文件中
while true
do
    # 计算内存占用率，并输出到终端和文件中
    mem_usage=$(free | awk '/Mem/{printf("%.2f%"), $3/$2*100}')
    current_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$current_time] 当前内存占用率为: $mem_usage"
    echo "[$current_time] 内存占用率: $mem_usage" >> $log_file
    
    # 等待1秒钟再继续执行循环
    sleep 1
done

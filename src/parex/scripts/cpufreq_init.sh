#!/bin/sh

for file in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
do
    echo performance > $file
done

# Intel
echo 0 > /proc/sys/kernel/randomize_va_space
echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo
echo 0 > /sys/devices/system/cpu/cpu9/online
# AMD
# echo 0 > /sys/devices/system/cpu/cpufreq/boost

cat /proc/cpuinfo | grep MHz

# https://llvm.org/docs/Benchmarking.html#linux
# https://vincent.bernat.ch/en/blog/2017-linux-kernel-microbenchmark

for file in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
do
    echo powersave > $file
done

# Intel
echo 1 > /proc/sys/kernel/randomize_va_space
echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo
echo 1 > /sys/devices/system/cpu/cpu9/online
# AMD
# echo 1 > /sys/devices/system/cpu/cpufreq/boost

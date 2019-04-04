#!/bin/bash

# initialize the parameters to 0
cs_tot=0
cs_avg=0
cs_min=0
cs_max=0

cp_tot=0
cp_avg=0
cp_min=0
cp_max=0

ap_tot=0
ap_avg=0
ap_min=0
ap_max=0

dp0_tot=0
dp0_avg=0
dp0_min=0
dp0_max=0

dp1_tot=0
dp1_avg=0
dp1_min=0
dp1_max=0

dp2_tot=0
dp2_avg=0
dp2_min=0
dp2_max=0

# loop through; execute; and sum
for((i=0; i < 32; i+=1)); do
	nvprof --log-file nvprof.log ./PGaB ../data/IRISC_dv4_R050_L54_N1296_Dform ../data/IRISC_dv4_R050_L54_N1296_Dform_Res 

	cs_tot=`python -c "print($cs_tot + $(cat nvprof.log | grep ComputeSyndrome | cut -b 28-33))"`
	cs_avg=`python -c "print($cs_avg + $(cat nvprof.log | grep ComputeSyndrome | cut -b 48-53))"`
	cs_min=`python -c "print($cs_min + $(cat nvprof.log | grep ComputeSyndrome | cut -b 58-63))"`
	cs_max=`python -c "print($cs_max + $(cat nvprof.log | grep ComputeSyndrome | cut -b 68-73))"`

	cp_tot=`python -c "print($cp_tot + $(cat nvprof.log | grep CheckPassGB | cut -b 28-33))"`
	cp_avg=`python -c "print($cp_avg + $(cat nvprof.log | grep CheckPassGB | cut -b 48-53))"`
	cp_min=`python -c "print($cp_min + $(cat nvprof.log | grep CheckPassGB | cut -b 58-63))"`
	cp_max=`python -c "print($cp_max + $(cat nvprof.log | grep CheckPassGB | cut -b 68-73))"`

	ap_tot=`python -c "print($ap_tot + $(cat nvprof.log | grep APP_GB | cut -b 28-33))"`
	ap_avg=`python -c "print($ap_avg + $(cat nvprof.log | grep APP_GB | cut -b 48-53))"`
	ap_min=`python -c "print($ap_min + $(cat nvprof.log | grep APP_GB | cut -b 58-63))"`
	ap_max=`python -c "print($ap_max + $(cat nvprof.log | grep APP_GB | cut -b 68-73))"`

	dp0_tot=`python -c "print($dp0_tot + $(cat nvprof.log | grep DataPassGB_0 | cut -b 28-33))"`
	dp0_avg=`python -c "print($dp0_avg + $(cat nvprof.log | grep DataPassGB_0 | cut -b 48-53))"`
	dp0_min=`python -c "print($dp0_min + $(cat nvprof.log | grep DataPassGB_0 | cut -b 58-63))"`
	dp0_max=`python -c "print($dp0_max + $(cat nvprof.log | grep DataPassGB_0 | cut -b 68-73))"`

	dp1_tot=`python -c "print($dp1_tot + $(cat nvprof.log | grep DataPassGB_1 | cut -b 28-33))"`
	dp1_avg=`python -c "print($dp1_avg + $(cat nvprof.log | grep DataPassGB_1 | cut -b 48-53))"`
	dp1_min=`python -c "print($dp1_min + $(cat nvprof.log | grep DataPassGB_1 | cut -b 58-63))"`
	dp1_max=`python -c "print($dp1_max + $(cat nvprof.log | grep DataPassGB_1 | cut -b 68-73))"`

	dp2_tot=`python -c "print($dp2_tot + $(cat nvprof.log | grep DataPassGB_2 | cut -b 28-33))"`
	dp2_avg=`python -c "print($dp2_avg + $(cat nvprof.log | grep DataPassGB_2 | cut -b 48-53))"`
	dp2_min=`python -c "print($dp2_min + $(cat nvprof.log | grep DataPassGB_2 | cut -b 58-63))"`
	dp2_max=`python -c "print($dp2_max + $(cat nvprof.log | grep DataPassGB_2 | cut -b 68-73))"`

	# for monitering temp
	sensors | grep "temp[0-9]"
	sensors | grep "Core [0-9]"
done

# divide all by loop count to get average
cs_tot=`python -c "print($cs_tot/32)" | cut -b 1-7`ms
cs_avg=`python -c "print($cs_avg/32)" | cut -b 1-7`us
cs_min=`python -c "print($cs_min/32)" | cut -b 1-7`us
cs_max=`python -c "print($cs_max/32)" | cut -b 1-7`us

cp_tot=`python -c "print($cp_tot/32)" | cut -b 1-7`ms
cp_avg=`python -c "print($cp_avg/32)" | cut -b 1-7`us
cp_min=`python -c "print($cp_min/32)" | cut -b 1-7`us
cp_max=`python -c "print($cp_max/32)" | cut -b 1-7`us

ap_tot=`python -c "print($ap_tot/32)" | cut -b 1-7`ms
ap_avg=`python -c "print($ap_avg/32)" | cut -b 1-7`us
ap_min=`python -c "print($ap_min/32)" | cut -b 1-7`us
ap_max=`python -c "print($ap_max/32)" | cut -b 1-7`us

dp0_tot=`python -c "print($dp0_tot/32)" | cut -b 1-7`ms
dp0_avg=`python -c "print($dp0_avg/32)" | cut -b 1-7`us
dp0_min=`python -c "print($dp0_min/32)" | cut -b 1-7`us
dp0_max=`python -c "print($dp0_max/32)" | cut -b 1-7`us

dp1_tot=`python -c "print($dp1_tot/32)" | cut -b 1-7`ms
dp1_avg=`python -c "print($dp1_avg/32)" | cut -b 1-7`us
dp1_min=`python -c "print($dp1_min/32)" | cut -b 1-7`us
dp1_max=`python -c "print($dp1_max/32)" | cut -b 1-7`us

dp2_tot=`python -c "print($dp2_tot/32)" | cut -b 1-7`ms
dp2_avg=`python -c "print($dp2_avg/32)" | cut -b 1-7`us
dp2_min=`python -c "print($dp2_min/32)" | cut -b 1-7`us
dp2_max=`python -c "print($dp2_max/32)" | cut -b 1-7`s

# Stream results to local file
echo "ComputeSyndrome" > stats.txt
echo "Total   : $cs_tot" >> stats.txt
echo "Average : $cs_avg" >> stats.txt
echo "Min     : $cs_min" >> stats.txt
echo "Max     : $cs_max" >> stats.txt
echo -e "\n" >> stats.txt

echo "CheckPassGB" >> stats.txt
echo "Total   : $cp_tot" >> stats.txt
echo "Average : $cp_avg" >> stats.txt
echo "Min     : $cp_min" >> stats.txt
echo "Max     : $cp_max" >> stats.txt
echo -e "\n" >> stats.txt

echo "APP_GB" >> stats.txt
echo "Total   : $ap_tot" >> stats.txt
echo "Average : $ap_avg" >> stats.txt
echo "Min     : $ap_min" >> stats.txt
echo "Max     : $ap_max" >> stats.txt
echo -e "\n" >> stats.txt

echo "DataPassGB_0" >> stats.txt
echo "Total   : $dp0_tot" >> stats.txt
echo "Average : $dp0_avg" >> stats.txt
echo "Min     : $dp0_min" >> stats.txt
echo "Max     : $dp0_max" >> stats.txt
echo -e "\n" >> stats.txt

echo "DataPassGB_1" >> stats.txt
echo "Total   : $dp1_tot" >> stats.txt
echo "Average : $dp1_avg" >> stats.txt
echo "Min     : $dp1_min" >> stats.txt
echo "Max     : $dp1_max" >> stats.txt
echo -e "\n" >> stats.txt

echo "DataPassGB_2" >> stats.txt
echo "Total   : $dp2_tot" >> stats.txt
echo "Average : $dp2_avg" >> stats.txt
echo "Min     : $dp2_min" >> stats.txt
echo "Max     : $dp2_max" >> stats.txt
echo -e "\n" >> stats.txt

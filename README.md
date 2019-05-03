# ece569 demo instructions

# Instructions to run serial (CPU) code:
Assuming you are in the directory where your project folder exists   
  
1:cd project/build/  
2:make serial  
3:qsub run_PGaB_serial.pbs  


# Instructions to run cuda code:
Assuming you are in teh directory where your project folder exists   
  
1: cd project  
2: module load cuda91/toolkit/9.1.85   
3: cd build   
5: make  
6: qsub run_PGaB.pbs  

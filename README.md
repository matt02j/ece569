# ece569

# Instructions to run serial code:
Assuming you are in teh directory where your project folder exists   
  
1:cd project/build/  
2:make serial  
3:qsub run_PGaB_serial.pbs  


# Instructions to run cuda code:
Assuming you are in teh directory where your project folder exists   
  
1: cd project  
2: module load cuda91/toolkit/9.1.85   
3: cd build 
4: make clean   
5: make  
6: qsub run_PGaB.pbs  

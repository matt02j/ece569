# ece569 demo instructions

1:while in the directory with project.tar.gz  
  Then run: tar -zxvf project.tar.gz  
2:Change the cd line in both of the PBS scripts located in project/build. They should be changed to the file location where the project directory is located followed by /project/build.  
(ex: cd ~jeremysears/ocelote/project/build)    

This will allow you to run the following codes  

# Instructions to run serial (CPU) code:
Assuming you are in the directory where your project directory exists   
  
1:cd project/build/  
2:make serial  
3:qsub run_PGaB_serial.pbs  


# Instructions to run cuda code:
Assuming you are in the directory where your project directory exists   
  
1: cd project  
2: module load cuda91/toolkit/9.1.85   
3: cd build   
5: make  
6: qsub run_PGaB.pbs  

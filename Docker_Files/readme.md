The docker images for task-1 and task-2 are availabe at following link;
https://qmulprod-my.sharepoint.com/personal/acw636_qmul_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Facw636%5Fqmul%5Fac%5Fuk%2FDocuments%2FLAScarQS2022


How run docker containers;



Step-1.  uplaod the docker images using.
             docker load --input team46testevaluation.tar 

step-2.      docker run -v C:\My_Data\sateg0\validation\viz_tresults\TASK_1:/input:ro -v :/output:ro -ti team46testevaluation

where, path to images = C:\My_Data\sateg0\validation\viz_tresults\TASK_1

step-3. docker cp 15ed16c01a79:/output C:\My_Data\sateg0\task_1_both_data\task1_2d\test  

where 15ed16c01a79 = container ID, and C:\My_Data\sateg0\task_1_both_data\task1_2d\test = path in local machine to save predictions 

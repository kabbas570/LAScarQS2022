The docker images for task-1 and task-2 are availabe at following link;
https://drive.google.com/drive/u/1/folders/1F2tB5J64GWn6lAunl3DR3Yk7U3mdLHS4


How run docker containers;

For task-1:

Step-1:   load --input team46testevaluation_task1.tar.gz
Step-2:   docker run -v path_to_input_images:/input:ro -v :/output:ro -ti team46testevaluation
Step-3:    docker cp container_ID:/output path_to_save_predictions


For task-2:

Step-1:   docker load --input team46testevaluation_task2.tar.gz
Step-2:   docker run -v path_to_input_images:/input:ro -v :/output:ro -ti team46testevaluation
Step-3:    docker cp container_ID:/output path_to_save_predictions


The image name for both tasks is team46testevaluation, however, the .tar.gz files have suffixes for each task i.e _task1.tar.gz and _task2.tar.gz.

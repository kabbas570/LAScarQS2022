### Sequential Segmentation of the Left Atrium and Atrial Scars Using a Multi-scale Weight Sharing Network and Boundary-based Processing
The paper has been accepted at LAScarQS 2022: Left Atrial and Scar Quantification & Segmentation Challenge
in conjunction with STACOM and MICCAI 2022 (Sep 18th, 2022, Singapore).
  https://zmiclab.github.io/projects/lascarqs22/submit.html
  
  ### Overview of propsoed-method
  
  Left atrial (LA) segmentation and quantification of atrial
scars have opened a path to automating Atrial Fibrillation (AF) diagnosis. This paper proposes a two-stage approach for sequential segmentation of the LA cavity and scars. Our Multi-scale Weight Sharing (MSWS) Network extracts features at multiple scales and is used for LA cavity segmentation. We also propose a Boundary2Patches method which performs segmentation of scars around the detected LA cavity boundary. The MSWS network learns a better representation of features through sharing weights across scales, and the Boundary2Patches method focuses on smaller scars constrained in the region around the LA cavity wall. On the challenge cohort (validation set), our method achieves an average Dice score of 0.938 and 0.558 for the LA cavity and scars segmentation of task 1, and a Dice score of 0.846 for LA cavity segmentation of task 2. 


### Sequential Segmentation of the Left Atrium and Atrial Scars Using a Multi-scale Weight Sharing Network and Boundary-based Processing
The paper has been accepted at LAScarQS 2022: Left Atrial and Scar Quantification & Segmentation Challenge
in conjunction with STACOM and MICCAI 2022 (Sep 18th, 2022, Singapore).
  https://zmiclab.github.io/projects/lascarqs22/submit.html
  
  ### Overview of propsoed-method
  
  Left atrial (LA) segmentation and quantification of atrial
scars have opened a path to automating Atrial Fibrillation (AF) diagnosis. This paper proposes a two-stage approach for sequential segmentation of the LA cavity and scars. Our Multi-scale Weight Sharing (MSWS) Network extracts features at multiple scales and is used for LA cavity segmentation. We also propose a Boundary2Patches method which performs segmentation of scars around the detected LA cavity boundary. The MSWS network learns a better representation of features through sharing weights across scales, and the Boundary2Patches method focuses on smaller scars constrained in the region around the LA cavity wall. On the challenge cohort (validation set), our method achieves an average Dice score of 0.938 and 0.558 for the LA cavity and scars segmentation of task 1, and a Dice score of 0.846 for LA cavity segmentation of task 2. 

![image](https://user-images.githubusercontent.com/56618776/187899764-ddf673cf-9e0c-4cfc-88a2-718b3638d302.png)

### Multi-Scale Weight Sharing Network

All the weight sharing stages must have the same number of kernels in each layer. In a conventional encoder-decoder architecture such as UNet, the
features are extracted from only a single scale of input and down-sampled by multiple factors; thus, the convolutional kernels only learn a single scale features
from the input space. Instead, we employ the concept of kernel sharing across several scales and make kernels capable of learning the same characteristics from
various input spaces. Furthermore, all kernels share the same parameters at each encoder stage; thus, the overall number of parameters in the architecture does
not increase, and the network benefits from convolving the same kernels with varying dimensions of incoming feature maps.
![image](https://user-images.githubusercontent.com/56618776/187899890-97526b0a-9d1d-48e1-b24d-e41ed17aad12.png)


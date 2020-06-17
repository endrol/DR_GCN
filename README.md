# DR_GCN brief info
## diabetic retinopathy grading using lesion correlation learned by Graph Convolution Network    
Network takes Diabetic Retinopathy fundus images as input, output the grading result.
<div align=center><img width="500" src="https://raw.githubusercontent.com/endrol/DR_GCN/master/dr_gcn/IMG/gradign%20(1).png"/>
<div align=left>
  
## Whole model structure shows like this.   

<div align=center><img width="700" src="https://raw.githubusercontent.com/endrol/DR_GCN/master/dr_gcn/IMG/strfinal.png"/>  
<div align=left>
We combine the lesion correlation graph learned by Graph Convolution Network (GCN), combined with CNN fundus image features, and do the grading. Into 5 grades.
SIFT extracted ROI vs SURF extracted ROI  
<div align=center><img width="500" src="https://raw.githubusercontent.com/endrol/DR_GCN/master/dr_gcn/IMG/sift_surfcom.png"/> 
<div align=left>

## lesion correlation graph constructed process
SURF features construct Nodes and their cooccurence construct edge information
<div align=center><img width="600" src="https://raw.githubusercontent.com/endrol/DR_GCN/master/dr_gcn/IMG/2%20(1).png"/>
<div align=left>
  
## experiment result
<div align=center><img width="700" src="https://raw.githubusercontent.com/endrol/DR_GCN/master/dr_gcn/IMG/roc_plot%20(1).png"/>
<div align=center><img width="700" src="https://raw.githubusercontent.com/endrol/DR_GCN/master/dr_gcn/IMG/confusion_matrix%20(1).png"/>
  <div align=left>
<img width="400" src="https://raw.githubusercontent.com/endrol/DR_GCN/master/dr_gcn/IMG/chart.png"/>

# Deploying details
The DR_GCN model definition can be found in [demo_dr_gcn.py](dr_gcn/demo_dr_gcn.py)
SURF extraction and clustering process to get the Nodes and Edge information can be found in [kmeans_feature_adj.py](dr_gcn/kmeans_feature_adj.py) and [surf_feature.py](dr_gcn/surf_feature.py)

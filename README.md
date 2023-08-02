# GACA
this is the code for the paper "GACA:A Gradient-Aware and Contrastive-Adaptive
Learning Framework for Low-Light Image Enhancement"

<img src="read/fig2.jpg">

### Dependencies and Installation
- Pytorch >= 1.13.1
- CUDA >= 11.3
- Other required packages in `requirements.txt`
```
# git clone this repository 
git clone https://github.com/iijjlk/GACA.git
cd GACA

# create new anaconda env
conda create -n GACA python=3.7 -y
conda activate GACA

# install python dependencies
pip install -r requirements.txt
```
<!-- ### Training: -->
###Testing

####Prepare Testing Data:

You can put the testing images in the test folder. Run the script below, 
the results will be saved in results folder.

```
python val.py 
```



### Contact
If you have any questions, please feel free to reach me out at `yaozishu@qdu.edu.cn`. 



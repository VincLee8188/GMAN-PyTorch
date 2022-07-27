# PyTorch implementation of GMAN: A Graph Multi-Attention Network for Traffic Prediction

# Important Note
In the original version of this repository the code was only utilizing CPU resulting in longer training times. This version is utilizes GPU and results in shorter learning times.

You can use the following command for training:

`python3 main.py --time_slot 5 --num_his 12 --num_pred 6 --batch_size 16 --max_epoch 150 --learning_rate 0.01`


This is a testing PyTorch version implementation of Graph Multi-Attention Network in the following paper: Chuanpan Zheng, Xiaoliang Fan, Cheng Wang, and Jianzhong Qi. "[GMAN: A Graph Multi-Attention Network for Traffic Prediction](https://arxiv.org/abs/1911.08415)", AAAI2020.

##  Requirements
* Python
* PyTorch
* Pandas
* Matplotlib
* Numpy

## Dataset

The datasets could be  unzipped and load from the data directory in this repository.




## Citation

This version of implementation is only for learning purpose. For research, please refer to  and  cite from the following paper:
```
@inproceedings{ GMAN-AAAI2020,
  author = "Chuanpan Zheng and Xiaoliang Fan and Cheng Wang and Jianzhong Qi"
  title = "GMAN: A Graph Multi-Attention Network for Traffic Prediction",
  booktitle = "AAAI",
  pages = "1234--1241",
  year = "2020"
}
```

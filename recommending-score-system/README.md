# Recommendation Scoring System


- 程序使用

  -svd 表示使用基于矩阵分解的预测方式（默认为用户协同过滤和商品协同过滤方式的预测），

  ```
  optional arguments:
    -h, --help            show this help message and exit
    -s SEED, --seed SEED  random seed for initializing matrix elements
    -n, --normal          if randomly initialize elements based on normal distribution
    -old                  if use the old parameters matrix that have existed
    -plot                 plot temporal phenomenon of scores
    -svd                  SVD matrix factorization (if "none", use memory-based prediction)
    -k K                  how many neighbors for prediction collaborative filtering
    -d DPATH, --dpath DPATH
                          folder path of output file
    -e EPOCH, --epoch EPOCH
                          number of epochs for optimization
    -r RATIO, --ratio RATIO
                          ratio of the origin train set splitted for valid set
    -lr LR                learning rate of SGD optimization
    -lam LAM              lambda: the regularization parameter
    -f F                  dim of hidden factors
  ```

  例如输入`python .\main.py -svd -f 5 -lam 0.001 -n -e 6`表示使用基于模型的预测方式、隐属性个数5，正则化参数0.001，正态分布初始化参数，训练6个epoch。

- 基于内存算法
  
  （参考实验报告, result/report.pdf）


- 基于模型算法

  （参考实验报告）


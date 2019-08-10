## Get criteo dataset

http://labs.criteo.com/2014/02/download-kaggle-display-advertising-challenge-dataset/


## Run benchmarks
```bash
python benchmark.py --model FM --factor_dim 16
python benchmark.py --model FFM --factor_dim 4
python benchmark.py --model FNN --factor_dim 16 --fnn_hidden 256,128,1
python benchmark.py --model AFM --factor_dim 16 --
```

### Results
| model | earlystop | loss  |  auc  |
|-------|-----------|-------|-------|
|  FM   |    27M    | .4553 | .7957 |
|  FFM  |    18M    | .4488 | .8021 |
|  FNN  |    18M    | .4546 | .7968 | 


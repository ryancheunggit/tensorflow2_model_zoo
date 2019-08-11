## Get criteo dataset

http://labs.criteo.com/2014/02/download-kaggle-display-advertising-challenge-dataset/


## Run benchmarks
```bash
python benchmark.py FM \
    --factor_dim 16
python benchmark.py FFM \
    --factor_dim 4
python benchmark.py FNN \
    --factor_dim 16 \
    --fc_hidden_sizes 256,128,1
python benchmark.py AFM \
    --factor_dim 16 \
    --attn_size 16
python benchmark.py DeepFM \
    --factor_dim 16 \
    --fc_hidden_sizes 256,128,1
python benchmark.py NFM \ 
    --factor_dim 64 \
    --fc_hidden_sizes 32,16,1
python benchmark.py xDeepFM \ 
    --factor_dim 16 \
    --fc_hidden_sizes 256,128,1 \
    --cin_hidden_sizes 128,128
python benchmark.py AFI \ 
    --factor_dim 16 \
    --attn_heads 2 \
    --attn_layers 3 \
    --fc_hidden_sizes 256,128,1
python benchmark.py FNFM \
    --factor_dim 4
    --fc_hidden_sizes 256,128,1
```

### Results
|  model  | earlystop | loss  |  auc  |
|---------|-----------|-------|-------|
|    FM   |    27M    | .4553 | .7957 |
|   FFM   |    18M    | .4488 | .8021 |
|   FNN   |    18M    | .4546 | .7968 | 
|   AFM   |    72M    | .4573 | .7923 |
|  DeepFM |    18M    | .4522 | .7983 |
|   NFM   |    18M    | .4535 | .7965 |
| xDeepFM |    18M    | .4523 | .7984 |
| AutoInt |    18M    | .4532 | .7970 |
|  FNFM   |     9M    | .4499 | .8017 |

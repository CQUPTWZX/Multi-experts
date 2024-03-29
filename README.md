# Uncertainty-guided Multi-Expert Learning for Long-tailed Hierarchical Text Classification

## Requirements

* Python >= 3.6
* torch >= 1.6.0
* transformers == 4.2.1
* fairseq >= 0.10.0
* torch-geometric == 1.7.2
* torch-scatter == 2.0.8
* torch-sparse ==  0.6.12

## Preprocess

Please download the original dataset. Transform your dataset to json format file {'token': List[str], 'label': List[str]}  
You can refer to datasets/preprocess.py

### Web Of Science (WOS)

The original dataset can be acquired freely in the repository of [HDLTex](https://github.com/kk7nc/HDLTex). Preprocess code could refer to the repository of [HiAGM](https://github.com/Alibaba-NLP/HiAGM). Please download the release of **WOS-46985(version 2)**.

### RCV1-V2

The preprocessing code could refer to the [repository of reuters_loader](https://github.com/ductri/reuters_loader) and we provide a copy here. The original dataset can be acquired [here](https://trec.nist.gov/data/reuters/reuters.html) by signing an agreement. It took us 1 data to receive a response.

### AAPD
They collect the abstract and the corresponding subjects of 55,840 papers in the computer science field from the Arxiv.[SGM](https://github.com/lancopku/SGM).

### BGC
BGC dataset is available at [Blurb Genre Collection](www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html).

## Train
```shell
usage: train.py [-h] [--lr LR] [--data {WebOfScience,nyt,rcv1}] [--batch BATCH] [--early-stop EARLY_STOP] [--device DEVICE] --name NAME [--update UPDATE] [--warmup WARMUP] [--contrast CONTRAST] [--graph GRAPH] [--layer LAYER]
                [--multi] [--lamb LAMB] [--thre THRE] [--tau TAU] [--seed SEED] [--wandb] [--experts] [--ta] [--eta]
```

| Option |  Description |
|--------|-------------|
| -h, --help | show this help message and exit |
| --lr LR | Learning rate |
| --data {WebOfScience,nyt,rcv1} | Dataset |
| --batch BATCH | Batch size |
| --early-stop EARLY_STOP | Epoch before early stop |
| --device DEVICE | cuda or cpu. Default: cuda |
| --name NAME | A name for different runs |
| --update UPDATE | Gradient accumulate steps |
| --warmup WARMUP | Warmup steps |
| --contrast CONTRAST | Whether use contrastive model. Default: True |
| --graph GRAPH | Whether use graph encoder. Default: True |
| --layer LAYER | Layer of Graphormer |
| --multi | Whether the task is multi-label classification. Should keep default since all datasets are multi-label classifications. Default: True |
| --lamb LAMB | lambda |
| --thre THRE | Threshold for keeping tokens. Denote as gamma in the paper. |
| --tau TAU | Temperature for contrastive model |
| --seed SEED | Random seed |
| --wandb | Use wandb for logging |
| --experts | Number of experts |
| --ta | If prefix weight ≤ tau , the loss of expert m on the sample will be eliminated. |
| --eta | Eta is a temperature factor that adjusts the sensitivity of prefix weights. |



e.g. Train on `WebOfScience` with `batch=16, lambda=0.05, gamma=0.02, experts=4, ta=0.5, eta=0.91`. Checkpoints will be in `checkpoints/WebOfScience-test/`.

```shell
python train.py --name test --batch 16 --data WebOfScience --lamb 0.05 --thre 0.02 --experts 4 --ta 0.5 --eta 0.91
```

### Reproducibility

We apply the BERT model as a text encoder. For Graphormer, we set the adaptive graph attention head to 8, the batch size to 16, and the feature size to 768. The selected optimizer is Adam, and the learning rate is set to 3×10−5 . The default training epoch is 20. If the effect does not improve after 5 epochs, the training is suspended in advance. The number of experts is set to 4, and the threshold τ = 0.91. Four experts are to jointly train tailed samples and one expert is used to train shallow-head samples. The threshold γ is set to 0.02 on WOS and AAPD, 0.005 on RCV1-V2, and 0.01 on BGC. Respectively. The loss weight λ is 0.1. Different batch sizes can have some impact on the model. We implement the model in PyTorch and experimented on NVIDIA GeForce RTX 3090.

## Test
```shell
usage: test.py [-h] [--device DEVICE] [--batch BATCH] [--name NAME] [--experts] [--eta]
```
| Option |  Description |
|--------|-------------|
| -h, --help | show this help message and exit |
| --device DEVICE | cuda or cpu. Default: cuda |
| --batch BATCH | Batch size |
| --name NAME | Name of checkpoint. Commonly as DATA-NAME |
| --experts | Number of experts |
| --eta | Eta is a temperature factor that adjusts the sensitivity of prefix weights. |



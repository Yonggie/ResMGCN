<h2 align="center">
<p> ResMGCN: Residual Message Graph Convolution Network for Fast Biomedical Interactions Discovering </h2>

Biomedical information graphs are crucial for interaction discovering of biomedical information in modern age, such as identification of multifarious molecular interactions and drug discovery, which attracts increasing interests in biomedicine, bioinformatics, and human healthcare communities. Nowadays, more and more graph neural networks have been proposed to learn the entities of biomedical information and precisely reveal biomedical molecule interactions with state-of-the-art results. These methods remedy the fading of features from a far distance but suffer from remedying such problem at the expensive cost of redundant memory and time. In our paper, we propose a novel Residual Message Graph Convolution Network (ResMGCN) for fast and precise biomedical interaction prediction in a different idea. Specifically, instead of enhancing the message from far nodes, ResMGCN aggregates lower-order information with the next round higher information to guide the node update to obtain a more meaningful node representation. ResMGCN is able to perceive and preserve various messages from the previous layer and high-order information in the current layer with least memory and time cost to obtain informative representations of biomedical entities. We conduct experiments on four biomedical interaction network datasets, including protein-protein, drug-drug, drug-target, and gene-disease interactions, which demonstrates that ResMGCN outperforms previous state-of-the-art models while achieving superb effectiveness on both storage and time. 



## Install

```bash
conda create --name env_name -f envs.yml
```

## Best param Example

See [best_param.sh](./best_param.sh).
A ResMGCN Jupyter notebook example is provided in [ResMGCN](./ResMGCN.ipynb), and SkipGNN Jupyter notebook examp9le is provided in [Example_Train](Example_Train.ipynb).


## Dataset

We provide the dataset in the [data](data/) folder. 

For details, please check [SkipGNN](https://github.com/kexinhuang12345/SkipGNN)

## Model
``ResidualGCN`` as model in ``SkipGNN/models.py`` and ``ResGCN`` as layer(module) in ``SkipGNN/layers.py``.
## Cite

Cite [arxiv](https://arxiv.org/abs/2311.07632) for now:

```
@misc{yin2023resmgcn,
      title={ResMGCN: Residual Message Graph Convolution Network for Fast Biomedical Interactions Discovering}, 
      author={Zecheng Yin},
      year={2023},
      eprint={2311.07632},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

The code framework is based on [SkipGNN](https://github.com/kexinhuang12345/SkipGNN).


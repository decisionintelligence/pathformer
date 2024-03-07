## (ICLR 2024) Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting

This code is a PyTorch implementation of our ICLR'24 paper "Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting". [[arXiv]](https://arxiv.org/abs/2402.05956)

🌟 Pathformer代码在阿里云仓库也进行同步更新：[阿里云Pathformer代码链接](https://github.com/alibaba/sreworks-ext/tree/main/aiops/Pathformer_ICLR2024)

## Citing Pathformer
If you find this resource helpful, please consider to cite our research:

```
@inproceedings{chen2024pathformer,
  author       = {Peng Chen and Yingying Zhang and Yunyao Cheng and Yang Shu and Yihang Wang and Qingsong Wen and Bin Yang and Chenjuan Guo},
  title        = {Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting},
  booktitle    = {International Conference on Learning Representations (ICLR)},
  year         = {2024}
}
```




## Introduction
 Pathformer, a Multi-Scale Transformer with Adaptive Pathways for time series forecasting. It integrates multi-scale temporal resolutions and temporal distances by introducing patch division with multiple patch sizes and dual attention on the divided patches, enabling the comprehensive modeling of multi-scale characteristics. Furthermore, adaptive pathways dynamically select and aggregate scale-specific characteristics based on the different temporal dynamics.


![The architecture of Pathformer](./figs/framework.png#pic_center)

The important components of Pathformer: Multi-Scale Transformer Block and Multi-Scale Router.

![The structure of the Multi-Scale Transformer Block and Multi-Scale Router](./figs/multi-scale%20transformer.png)
## Requirements
To install all dependencies:
```
pip install -r requirements.txt
```

## Datasets
You can access the well pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP/view), then place the downloaded contents under ./dataset
## Quick Demos
1. Download datasets and place them under ./dataset
2. Run each script in scripts/, for example
```
bash scripts/multivariate/ETTm2.sh
```


## Further Reading
1, [**Transformers in Time Series: A Survey**](https://arxiv.org/abs/2202.07125), in IJCAI 2023.
[\[GitHub Repo\]](https://github.com/qingsongedu/time-series-transformers-review)

```bibtex
@inproceedings{wen2023transformers,
  title={Transformers in time series: A survey},
  author={Wen, Qingsong and Zhou, Tian and Zhang, Chaoli and Chen, Weiqi and Ma, Ziqing and Yan, Junchi and Sun, Liang},
  booktitle={International Joint Conference on Artificial Intelligence(IJCAI)},
  year={2023}
}
```



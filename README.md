# Cross-Modality Modeling for Time Series Analytics

This repository contains the code for our IJCAI'25 [paper](https://arxiv.org/abs/2505.02583) "*Towards Cross-Modality Modeling for Time Series Analytics: A Survey in the LLM Era*", where we propose a taxonomy of cross-modal time series analytics and further investigate what kind of text, how, and when the text modality can help LLM for time series forecasting.

## Dependencies

```bash
conda create --name tsf python=3.11
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers
pip install reformer_pytorch
pip install sktime
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install tqdm
```

## Data Preprocess

We provide datasets, which you can access [here](https://drive.google.com/file/d/1NqFkzcIiQnQqaA5OyXJ4XhHM3KiV89Vb/view?usp=drive_link). 

<!-- We download the numerical and text data into `numerical` and `textual` sub-folders under `data` folder. -->

We combine the numerical and text data into a single csv file using `data_analysis/data_align.py`.

We prepare examples in `data_analysis/run_data_prep.sh`.

## Training

We use the scripts in `scripts` folder to conduct the experiments.

The naming convention is {frequency}_{dataset}_{fusion_method}.

## BibTex
> If you find our work useful in your research. Please consider giving a star ⭐ and citation 📚:
```bibtex
@inproceedings{liu2025cm2ts,
  title={Towards Cross-Modality Modeling for Time Series Analytics: A Survey in the LLM Era},
  author={Chenxi Liu, Shaowen Zhou, Qianxiong Xu, Hao Miao, Cheng Long, Ziyue Li, Rui Zhao},
  booktitle    = {IJCAI},
  year={2025}
}
```

## Further Reading

[**Spatial-Temporal Large Language Model for Traffic Prediction**](https://github.com/ChenxiLiu-HNU/ST-LLM/blob/main/ST-LLM.pdf), in *IJCAI* 2025.
[\[GitHub Repo\]](https://github.com/ChenxiLiu-HNU/ST-LLM)

**Authors**: Chenxi Liu, Shaowen Zhou, Qianxiong Xu, Hao Miao, Cheng Long, Ziyue Li, Rui Zhao

```bibtex
@inproceedings{liu2024spatial,
  title={Spatial-Temporal Large Language Model for Traffic Prediction},
  author={Liu, Chenxi and Yang, Sun and Xu, Qianxiong and Li, Zhishuai and Long, Cheng and Li, Ziyue and Zhao, Rui},
  booktitle={MDM},
  year={2024}
}
```

[**ST-LLM+: Graph Enhanced Spatio-Temporal Large Language Models for Traffic Prediction**](https://www.computer.org/csdl/journal/tk/5555/01/11005661/26K27tC6ki4), in *TKDE* 2025.
[\[GitHub Repo\]](https://github.com/kethmih/ST-LLM-Plus)

**Authors**: Chenxi Liu, Kethmi Hirushini Hettige, Qianxiong Xu, Cheng Long, Shili Xiang, Gao Cong, Ziyue Li, Rui Zhao

```bibtex
@article{liu2025stllm_plus,
  title={{ST-LLM+}: Graph Enhanced Spatial-Temporal Large Language Model for Traffic Prediction},
  author={Chenxi Liu and  Hettige Kethmi Hirushini and Qianxiong Xu and Cheng Long and Ziyue Li and Shili Xiang and Rui Zhao and Gao Cong},
  journal    = {{IEEE} Transactions Knowledge Data Engineering},
  pages      = {1-14},
  year={2025}
}
```

[**TimeCMA: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment**](https://arxiv.org/abs/2406.01638), in *AAAI* 2025.
[\[GitHub Repo\]](https://github.com/ChenxiLiu-HNU/TimeCMA)

**Authors**: Chenxi Liu, Qianxiong Xu, Hao Miao, Sun Yang, Lingzheng Zhang, Cheng Long, Ziyue Li, Rui Zhao

```bibtex
@inproceedings{liu2024timecma,
  title={{TimeCMA}: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment},
  author={Liu, Chenxi and Xu, Qianxiong and Miao, Hao and Yang, Sun and Zhang, Lingzheng and Long, Cheng and Li, Ziyue and Zhao, Rui},
  booktitle={AAAI},
  year={2025}
}
```

[**Efficient Multivariate Time Series Forecasting via Calibrated Language Models with Privileged Knowledge Distillation**](https://arxiv.org/abs/2505.02138), in *ICDE* 2025.
[\[GitHub Repo\]](https://github.com/ChenxiLiu-HNU/TimeKD)

**Authors**: Chenxi Liu, Hao Miao, Qianxiong Xu, Shaowen Zhou, Cheng Long, Yan Zhao, Ziyue Li, Rui Zhao

```bibtex
@inproceedings{liu2025timekd,
  title={Efficient Multivariate Time Series Forecasting via Calibrated Language Models with Privileged Knowledge Distillation},
  author={Chenxi Liu and Hao Miao and Qianxiong Xu and Shaowen Zhou and Cheng Long and Yan Zhao and Ziyue Li and Rui Zhao},
  booktitle    = {ICDE},
  year={2025}
}
```

## Contact
If you have any questions or suggestions, feel free to contact shaowen310@outlook.com and chenxi.liu@ntu.edu.sg.

## Acknowledgement

This library is constructed based on the following repos:

[https://github.com/AdityaLab/MM-TSFlib](https://github.com/AdityaLab/MM-TSFlib)

# Cross-modal Time Series Forecasting

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

We provide preprocessed datasets, which you can access [here](https://drive.google.com/file/d/1NqFkzcIiQnQqaA5OyXJ4XhHM3KiV89Vb/view?usp=drive_link). 

<!-- We download the numerical and text data into `numerical` and `textual` sub-folders under `data` folder. -->

We combine the numerical and text data into a single csv file using `data_analysis/data_align.py`.

We prepare examples in `data_analysis/run_data_prep.sh`.

## Training

We use the scripts in `scripts` folder to conduct the experiments.

The naming convention is {frequency}_{dataset}_{fusion_method}.

## Contact
If you have any questions or suggestions, feel free to contact:
shaowen310@outlook.com

## Acknowledgement

This library is constructed based on the following repos:

[https://github.com/AdityaLab/MM-TSFlib](https://github.com/AdityaLab/MM-TSFlib)

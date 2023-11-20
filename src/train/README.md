### This dir contains datasets and source code in Jupyter notebook to train model

#### Steps:
- install dependencies: `numpy pandas fasttext` using `pip` or `poetry` package managers
- join and unzip dataset:
    ```
    cat src/train/train_val_data_a* > train_val_data.zip
    unzip train_val_data.zip
    ```
- run `full_training.ipynb` notebook
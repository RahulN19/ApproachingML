import config
import argparse
import os

import pandas as pd
from sklearn.model_selection import KFold

def run(fold_size):

    # Read the training file
    df_train = pd.read_csv(config.TRAINING_FILE_NAME)

    # create a new column called kfold and initialize to -1
    df_train.loc[:, "kfold"] = -1

    kfold = KFold(fold_size)

    # Shuffling the data
    df_train.sample(frac=1).reset_index(drop=True)

    for fold, (train_,val_) in enumerate(kfold.split(df_train)):
        df_train.loc[val_, "kfold"] = fold

    df_train.to_csv(os.path.join(config.INPUT_FILE_LOCATION,"train_data_folds.csv"),index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_size", type=int)
    args = parser.parse_args()
    run(args.fold_size)
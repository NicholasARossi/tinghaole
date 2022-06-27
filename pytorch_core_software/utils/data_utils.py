from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import logging
import pytz
import pathlib
import os
from datetime import datetime
import pickle
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


def save_predictions(folder, prediction_results):
    prediction_name = f'{get_date()}_predictions_results.pkl'
    full_path = os.path.join(folder, prediction_name)
    with open(full_path, 'wb') as f:
        pickle.dump(prediction_results, f)


def log_to_file(args, folder, filename):
    p = pathlib.Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(folder, filename), 'w') as f:
        if isinstance(args, list):
            for item in args:
                f.write(f"{item}\n")
        else:
            json.dumps(args.__dict__, f, indent=3)

    return


def get_date():
    date_format = '%m-%d-%Y'
    tz = pytz.timezone('US/Pacific')
    curr_date = datetime.now(tz).strftime(date_format)
    return curr_date


def get_datasets(input_sheet, target_feature='tone'):
    df = pd.read_csv(input_sheet)
    target_feature_types = ['tone', 'phoneme_encoding']
    if target_feature not in target_feature_types:
        raise ValueError(f"Invalid target feture types. Expected one of: {target_feature_types}")
    logger.info(f"Target Feature is : {target_feature}")
    logger.info(f"Number of Unique {target_feature}s in dataset is : {len(df[target_feature].unique())}")

    select_all_data_df = df[['absolute_file_path', target_feature]].rename(columns={target_feature: 'target_feature'})

    df_train, df_remainder = train_test_split(select_all_data_df, test_size=0.3)
    df_test, df_val = train_test_split(df_remainder, test_size=0.3)

    logger.info(f'Training data size : {len(df_train)}')
    logger.info(f'Testing data size : {len(df_test)}')
    logger.info(f'Validation data size : {len(df_val)}')
    return df_train, df_test, df_val


def get_callbacks(checkpoint_save_folder, patience):
    lr_monitor = LearningRateMonitor(logging_interval='step')

    monitor_val = 'val/f1_0'
    mode = 'max'

    checkpoint_val_MSE = ModelCheckpoint(
        monitor=monitor_val,
        dirpath=checkpoint_save_folder,
        filename='tinghaole:{epoch:02d}--val_F1:{val_F1:.3f}',
        save_top_k=1,
        mode=mode,
        auto_insert_metric_name=False
    )
    early_stopping_callback = EarlyStopping(monitor=monitor_val, patience=patience, verbose=True, mode=mode)
    return (lr_monitor, checkpoint_val_MSE, early_stopping_callback)

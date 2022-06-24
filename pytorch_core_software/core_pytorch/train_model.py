from pytorch_core_software.utils.data_utils import get_datasets,get_date,get_callbacks
from pytorch_core_software.core_pytorch.modules import CnnModule, DataModule
import pytorch_lightning as pl
import argparse
import pickle
import os

PATIENCE = 100


def main(args):
    df_train, df_test, df_val = get_datasets(args.data_path,
                                            target_feature=args.target_feature)

    # save config
    checkpoint_save_folder = f'model_outputs/{args.experiment_name}_{get_date()}'

    # get callbacks
    (lr_monitor, checkpoint_val, early_stopping_callback) = get_callbacks(checkpoint_save_folder,
                                                                          patience=PATIENCE)
    #
    # # determine module
    model_dict = {'cnn': CnnModule}
    #
    module = model_dict[args.model_type]
    #
    # if args.num_gpus == 0:
    # no GPU signifies debugging mode
    module = module(cuda=False)

    trainer = pl.Trainer(callbacks=[checkpoint_val, early_stopping_callback, lr_monitor])
    n_workers = 0

    datamodule = DataModule(df_train,
                            df_test,
                            df_val,
                            batch_size=args.batch_size,
                            n_workers=n_workers,
                            data_type=args.data_type)


    #
    # else:
    #     module = module(cuda=True, classification=args.classification, pos_weight=df_train.mean()['target_feature'])
    #
    #     trainer = pl.Trainer(gpus=args.num_gpus,
    #                          auto_select_gpus=True,
    #                          accelerator='ddp',
    #                          precision=16,
    #                          profiler='simple',
    #                          callbacks=[checkpoint_val, early_stopping_callback, lr_monitor])
    #
    #     n_workers = os.cpu_count()
    #

    #
    trainer.logger._version = checkpoint_save_folder
    #
    trainer.fit(module, datamodule)
    test_metrics = trainer.test(datamodule=datamodule)
    log_to_file(test_metrics, checkpoint_save_folder, 'test_metrics.txt')
    #
    prediction_results = trainer.predict(module, datamodule)
    #
    # with open('predictions_results.pkl', 'wb') as f:
    #     pickle.dump(prediction_results, f)

    pass
    # return test_metrics


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--data_path', type=str, help='path to dataset (CSV)', required=True,
                   default='')
    p.add_argument('--target_feature', type=str, default='tone', help='the target feature that we chose to analyze')
    p.add_argument('--experiment_name', type=str, help='name of the experiment', required=True, default='cnn_model')

    p.add_argument('--model_type', type=str, help='name of the model architecture', required=False, default='cnn')
    p.add_argument('--data_type', type=str, help='conversion scheme for the data encoding', required=False, default='msg')

    # p.add_argument('--num_gpus', type=int, required=True)
    p.add_argument('--batch_size', type=int, required=False, default=16)
    # p.add_argument('--target_feature', type=str, default='quality_score', help='name of the experiment')
    # p.add_argument('--classification', help='convert to classification problem', action='store_true')
    args = p.parse_args()
    main(args)

import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
import librosa
from torchmetrics.classification.stat_scores import StatScores
from transformers import AutoFeatureExtractor, AutoModelForPreTraining

import warnings

warnings.filterwarnings(action='ignore', module='librosa')


class DataSet(torch.utils.data.Dataset):
    DATA_MAX_DIMS = (80, 60)

    def __init__(self, df, data_type='msg', model_type='cnn'):
        self.data_locs = df['absolute_file_path']
        self.labels = df['target_feature'] - 1
        self.data_type = data_type
        self.cache_encodings = True
        self.encoding_lookup = {}
        self.model_type = model_type
        if model_type == 'transformer':
            self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    def __len__(self):
        return len(self.data_locs)

    def __getitem__(self, idx):

        label = self.labels.iloc[idx]
        if self.data_type == 'msg':
            if self.data_locs.iloc[idx] in self.encoding_lookup:
                spectrogram = self.encoding_lookup[self.data_locs.iloc[idx]]
            else:
                spectrogram = self.mp3toMSG(self.data_locs.iloc[idx])

                # cnn requires one extra dimension on input (filter)
                if self.model_type == 'cnn':
                    spectrogram = np.expand_dims(spectrogram, axis=0)
                if self.cache_encodings:
                    self.encoding_lookup[self.data_locs.iloc[idx]] = spectrogram

            return {'x': torch.from_numpy(spectrogram).type('torch.FloatTensor'),
                    'y': torch.from_numpy(np.asarray(label)).type('torch.LongTensor')}
        elif self.data_type == 'transform_array':
            audio, sample_rate = librosa.core.load(self.data_locs.iloc[idx])
            audio_resampled = librosa.resample(audio, orig_sr=sample_rate,
                                               target_sr=self.feature_extractor.sampling_rate)
            inputs = self.feature_extractor(
                audio_resampled, sampling_rate=self.feature_extractor.sampling_rate, max_length=16000, truncation=True,
            )
            filled_array = np.zeros((16000,))
            values = inputs['input_values'][0]
            filled_array[0:len(values)] = values

            return {'x': filled_array,
                    'y': torch.from_numpy(np.asarray(label)).type('torch.LongTensor')}
        else:
            raise ValueError(f"Invalid datatype conversion. {self.data_type} not supported")

        # todo check that this is the right type for classification

    @staticmethod
    def mp3tomfcc(file_path):
        """
        This method
        :param file_path:
        :return:
        """
        audio, sample_rate = librosa.core.load(file_path)

        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60)
        return mfcc

    def mp3toMSG(self, file_path, trimming=True) -> np.ndarray:
        """
        This method coverts to mels spectrogram

        :param file_path:
        :param trimming:
        :return:
        """
        audio, sample_rate = librosa.core.load(file_path)

        if trimming == True:
            audio = librosa.effects.trim(audio, top_db=20, frame_length=256, hop_length=64)[0]

        MSG = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=1024, hop_length=512, n_mels=80, fmin=75,
                                             fmax=3700)
        MSG = np.log10(MSG + 1e-10)

        # normalize
        MSG = (MSG - np.min(MSG)) / np.ptp(MSG)

        padded_data = self.add_padding(MSG, bonus_padding=2, maxes=self.DATA_MAX_DIMS)

        return padded_data

    @staticmethod
    def add_padding(array, bonus_padding=10, maxes=None):

        max_height, max_width = maxes[0], maxes[1]

        pad_width = (max_width - array.shape[1]) + bonus_padding
        pad_height = (max_height - array.shape[0]) + bonus_padding

        new_array = np.pad(array, pad_width=((bonus_padding, pad_height), (bonus_padding, pad_width)), mode='constant')

        return new_array


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 df_train,
                 df_test,
                 df_val,
                 batch_size=256,
                 pin_memory=True,
                 n_workers=20,
                 data_type='msg',
                 model_type='cnn'):
        super().__init__()
        self.train_dataset = DataSet(df_train, data_type=data_type, model_type=model_type)
        self.test_dataset = DataSet(df_test, data_type=data_type, model_type=model_type)
        self.val_dataset = DataSet(df_val, data_type=data_type, model_type=model_type)
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.n_workers = n_workers

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.n_workers,
                                           pin_memory=self.pin_memory)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.n_workers,
                                           pin_memory=self.pin_memory)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.n_workers,
                                           pin_memory=self.pin_memory)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.n_workers,
                                           pin_memory=self.pin_memory)


class CnnModule(pl.LightningModule):
    def __init__(self,
                 lr=1e-6,
                 patience=20,
                 target_feature='Tone',
                 model_type='cnn'):

        super().__init__()
        self.cnv1 = nn.Conv2d(1, 32, 2, padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.mxpool = nn.MaxPool2d(4)
        self.dropout1 = nn.Dropout(p=0.25)

        self.cnv2 = nn.Conv2d(32, 48, 2, padding='same')
        self.bn2 = nn.BatchNorm2d(48)

        self.maxpool2 = nn.MaxPool2d(4)
        self.dropout2 = nn.Dropout(p=0.1)
        self.cnv3 = nn.Conv2d(48, 120, 2, padding='same')
        self.bn3 = nn.BatchNorm2d(120)

        self.flatten = nn.Flatten()

        # self.cnv = nn.Conv2d(1, 64, kernel_size1, padding='same')
        self.lr = lr
        self.patience = patience

        self.fc1 = nn.Linear(645120, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_tone = nn.Linear(64, 4)
        self.fc_phoneme = nn.Linear(64, 100)

        self.target_feature = target_feature

        self.stats_scores = StatScores(num_classes=4, multiclass=True)

    def loss_fn(self, out, target):

        return nn.NLLLoss()(input=out, target=target)

        # return nn.CrossEntropyLoss()(input=out, target=target)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        monitor_val = 'val/f1_0'
        mode = 'max'

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  mode,
                                                                  patience=self.patience,
                                                                  factor=0.9,
                                                                  verbose=True)
        # todo double check labeling of loss functions
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': monitor_val
        }

        return [optimizer], [scheduler]

    def forward(self, x, predict=False):

        # first layer of convolutions
        out = self.bn1(self.relu(self.cnv1(x)))
        # Second layer of convolutions
        out = self.bn2(self.relu(self.cnv2(out)))
        # Third layer of convolutions
        out = self.bn3(self.relu(self.cnv3(out)))

        # max pool
        # out = self.maxpool2(out)
        # flatten
        out = self.flatten(out)

        # fully connected layers
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))

        if self.target_feature == 'Tone':
            out = self.fc_tone(out)

        else:
            raise ValueError(f"{self.target_feature} not supported")

        # multiclass softmax
        out = torch.softmax(out, axis=1)

        return out

    def _step(self, batch, prefix):
        inputs = batch["x"]
        labels = batch["y"]

        predictions = self.forward(inputs)
        output_loss = self.loss_fn(predictions, labels)
        logs = {f'loss/{prefix}_loss': output_loss}
        self.log_dict(logs)

        # update metrics
        if prefix != "train":
            _, y_pred_tags = torch.max(predictions, dim=1)
            self.stats_scores.update(y_pred_tags, labels)

        return output_loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, prefix='train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, prefix='val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, prefix='test')

    def predict_step(self, batch, batch_idx):
        dna_input_ids = batch["x"]
        labels = batch["y"]

        predictions = self.forward(dna_input_ids).squeeze()
        return predictions, labels

    def training_epoch_end(self, outputs):
        return

    def validation_epoch_end(self, outputs):
        self.compute_metrics('val')

    def test_epoch_end(self, outs):
        self.compute_metrics('test')

    def compute_metrics(self, prefix):

        on_epoch = None
        if prefix == 'train':
            on_epoch = True

        tp, fp, tn, fn, _ = self.stats_scores.compute()
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        precision_1 = tp / (tp + fp)
        recall_1 = tp / (tp + fn)
        f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)
        precision_0 = tn / (tn + fn)
        recall_0 = tn / (tn + fp)
        f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0)
        if torch.isnan(f1_0):
            f1_0 = 0
        if torch.isnan(f1_1):
            f1_1 = 0

        metric_dict = {
            f'{prefix}/accuracy': accuracy,
            f'{prefix}/precision_1': precision_1,
            f'{prefix}/recall_1': recall_1,
            f'{prefix}/f1_1': f1_1,
            f'{prefix}/precision_0': precision_0,
            f'{prefix}/recall_0': recall_0,
            f'{prefix}/f1_0': f1_0
        }

        self.log_dict(metric_dict, prog_bar=True, on_epoch=on_epoch)

        self.stats_scores.reset()


class LstmModule(CnnModule):
    def __init__(self,
                 lr=1e-3,
                 patience=20):
        super().__init__(lr=lr,
                         patience=patience)
        self.lstm = nn.LSTM(64, 32)
        self.lstm_dense = nn.Linear(2688, 128)

        self.lstm_dense2 = nn.Linear(128, 64)

    def forward(self, x, predict=False):
        out = self.lstm(x)[0]
        out = self.flatten(out)
        out = self.relu(self.lstm_dense(out))
        out = self.dropout1(out)
        out = self.relu(self.lstm_dense2(out))
        out = self.dropout2(out)
        if self.target_feature == 'Tone':
            out = self.fc_tone(out)

        else:
            raise ValueError(f"{self.target_feature} not supported")

        # multiclass softmax
        out = torch.softmax(out, axis=1)

        return out


class CnnLstmModule(CnnModule):
    pass


class AudioTransformer(CnnModule):
    def __init__(self,
                 lr=1e-3,
                 patience=20):
        super().__init__(lr=lr,
                         patience=patience)
        self.transformer_block = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-base")

    def forward(self, x, predict=False):
        out = self.transformer_block(input_values={'intput_values': x})

        if self.target_feature == 'Tone':
            out = self.fc_tone(out)

        else:
            raise ValueError(f"{self.target_feature} not supported")

        # multiclass softmax
        out = torch.softmax(out, axis=1)

        return out

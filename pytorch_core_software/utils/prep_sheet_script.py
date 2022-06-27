import os
from glob import glob
from pytorch_core_software.config.config_params import phoneme_categories
import argparse
import pandas as pd

def create_sheet(csv_name,
                 mp3_directory,
                 ma_only=False):

    files = glob(os.path.join(mp3_directory, '*.mp3'))
    substring_list = ['ma1', 'ma2', 'ma3', 'ma4']

    file_mapping_dict = {}
    index = 0
    for file in files:
        if not ma_only or (ma_only and any(map(file.__contains__, substring_list))):
            # full labels
            full_label = file.split('/')[-1].replace('_MP3.mp3', '')
            label = full_label.split('_')[0]
            absolute_file_path = os.path.abspath(file)
            tone = int(label[-1])
            phoneme = label[:-1]
            phoneme_encoding = phoneme_categories.index(phoneme)
            file_mapping_dict[index] = {'full_label': full_label,
                                        'label': label,
                                        'absolute_file_path': absolute_file_path,
                                        'tone': tone,
                                        'phoneme': phoneme,
                                        'phoneme_encoding': phoneme_encoding}

            index += 1
    pd.DataFrame(file_mapping_dict).T.to_csv(csv_name)


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--mp3_folder_path', type=str, help='path_to_mp3_fles', required=True, default='')
    p.add_argument('--csv_name', type=str, help='name, dir of output csv', required=True, default='')
    p.add_argument('--ma_only', action='store_true', required=False)

    args = p.parse_args()
    create_sheet(args.csv_name,
                 args.mp3_folder_path,
                 ma_only=args.ma_only)

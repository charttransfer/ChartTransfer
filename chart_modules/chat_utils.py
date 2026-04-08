import base64
from datetime import datetime
import json
import os
import time
import logging
from pathlib import Path
from PIL import Image


my_logger = None

def get_logger(name=None, log_path=None):
    global my_logger
    if my_logger is not None:
        return my_logger
    assert name is not None, 'name should not be None'
    assert log_path is not None, 'log_path should not be None'
    my_logger = logging.getLogger(name)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', \
        handlers=[logging.FileHandler(filename=log_path, encoding='utf-8', mode='a+'), logging.StreamHandler()])
    return my_logger

def load_json(save_path, output=False):
    info_dict = {}
    if os.path.exists(save_path):
        with open(save_path, "r", encoding='utf-8') as f:
            info_dict = json.load(f)
    if output:
        print('already have', len(info_dict))
    return info_dict


def load_txt(save_path):
    assert os.path.exists(save_path), f'{save_path} not exist'
    info = ''
    if os.path.exists(save_path):
        with open(save_path, "r", encoding='utf-8') as f:
            info = f.read()
    return info


def safe_save_json(info_dict, save_path, output=False):
    while True:
        try:
            with open(save_path, "w", encoding='utf-8') as f:
                json.dump(info_dict, f, indent=2, ensure_ascii=False)
            break
        except Exception as e:
            time.sleep(1)
            print('----------save error:', str(e))
            print('----------do not interrupt saving, retrying...')
    if output:
        print(f'--------------------save success,', len(info_dict), 'saved')
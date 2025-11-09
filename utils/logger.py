import logging
import os
import sys
import os.path as osp
from pathlib import Path
import re
from termcolor import colored


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def setup_logger(name, save_dir, if_train, color_text=True, time_mode=''):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    # formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

    plain_formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
    if color_text:
        formatter = _ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S",
            root_name=name,
            abbrev_name=str(name),
        )
    else:
        formatter = plain_formatter

    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        save_dir = Path(save_dir).resolve()
        exit_files = list(save_dir.rglob("train_log*.txt")) if if_train else list(save_dir.rglob("test_log*.txt"))
        exit_files_number = [0]
        for filename in exit_files:
            match = re.match(r"\w+_log_(\d*)", filename.stem)
            if match:
                number = int(match.group(1))  # 捕获的数字部分
            else:
                number = 0
            exit_files_number.append(number)

        log_file_idx = max(exit_files_number) + 1
        fpath = save_dir / f'train_log_{log_file_idx}.txt' if if_train else save_dir / f'test_log_{log_file_idx}.txt'

        fh = logging.FileHandler(fpath, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger

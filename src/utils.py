import json
from dataclasses import dataclass, asdict
import string
import os

wd = os.path.dirname(os.path.realpath(__file__))

DEFAULT_CONFIG_PATH = f"{wd}/../config/config.json"


def get_providers(path):
    if path == "DEFAULT" or path is None:
        path = DEFAULT_CONFIG_PATH

    with open(path, "r") as f:
        all_config = json.load(f)
        keys = all_config["csv"].keys()
    return list(keys)


def load_config(path, type=None):
    if path == "DEFAULT" or path is None:
        path = DEFAULT_CONFIG_PATH

    with open(path, "r") as f:
        all_config = json.load(f)
        wb_config = WbConfig(**all_config["tracker"])

        if type is not None:
            try:
                config = all_config["csv"]["base"]
            except KeyError:
                config = {}
                print(f"Couldn't find default config in {path}")
            try:
                config.update(all_config["csv"][type])
            except KeyError:
                print(f"Config type specified ({type}) does not exist in {path}")
                exit(1)

            # if fields are passed as letters convert them to numbers
            for key, value in config.items():
                if "field" in key:
                    if isinstance(config[key], list):
                        # If input is a list parse independantly
                        config[key] = [to_num(i) for i in config[key]]
                    else:
                        config[key] = to_num(value)
            csv_config = CsvConfig(source=type, **config)
        else:
            csv_config = None

    return GlobalConfig(csv_config, wb_config)


def load_labels(fp):
    if fp == "DEFAULT" or fp is None:
        fp = f"{wd}/../config/labels.txt"

    with open(fp, "r") as f:
        out = []
        for line in f:
            out.append(line.strip("\n").strip())
    return out


@dataclass
class Config:
    def to_dict(self):
        return asdict(self)


@dataclass
class CsvConfig(Config):
    source: str
    ammount_is_negative: bool
    val_field: int
    desc_field: list
    date_field: int
    date_frmt: str
    has_header_row: bool
    skip_words: list


@dataclass
class WbConfig(Config):
    date_col: str
    desc_col: str
    cat_col: str
    val_col: str
    source_col: str
    start_row: int
    title_cell: str
    data_sheet: str

    def to_dict(self):
        return asdict(self)


@dataclass
class GlobalConfig(Config):
    csv: CsvConfig
    wb: WbConfig

    def to_dict(self):
        return asdict(self)


def to_num(x):
    if isinstance(x, int):
        return x
    elif isinstance(x, str):
        if x.isdigit():
            return int(x)
        if x.lower() in string.ascii_lowercase:
            return string.ascii_lowercase.find(x.lower())

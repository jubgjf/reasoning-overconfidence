import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Type, assert_never

from pydantic import BaseModel
from typing_extensions import TypeVar

from .data import Data, GSM8KData, ARCData, Record, GSM8KRecord, ARCRecord, LogiQARecord, LogiQAData


class DatasetName(Enum):
    GSM8K = "gsm8k"
    ARC = "arc"
    LogiQA = "logiqa"

    def __str__(self) -> str:
        return self.value

    @property
    def record_cls(self) -> Type[Record]:
        if self == self.GSM8K:
            return GSM8KRecord
        elif self == self.ARC:
            return ARCRecord
        elif self == self.LogiQA:
            return LogiQARecord
        else:
            assert_never(self)

    @property
    def dataset_cls(self) -> Type["Dataset"]:
        if self == self.GSM8K:
            return GSM8KDataset
        elif self == self.ARC:
            return ARCDataset
        elif self == self.LogiQA:
            return LogiQADataset
        else:
            assert_never(self)


class IDataset(BaseModel, ABC):
    @property
    @abstractmethod
    def _name(self) -> DatasetName: ...

    @property
    @abstractmethod
    def _data_cls(self) -> Type[Data]: ...

    @property
    def _data_file(self) -> Path:
        return Path(f"./dataset/{self._name.value}.jsonl")

    def _load_full_dataset(self) -> list[Data]:
        with open(self._data_file) as f:
            dataset = [json.loads(line) for line in f]
        return [self._data_cls(**data) for data in dataset]

    def load_resume_dataset(self, already_processed_ids: list[int] | None, force_restart: bool = False) -> list[Data]:
        dataset = self._load_full_dataset()
        if force_restart:
            return dataset
        return [data for data in dataset if data.id not in already_processed_ids]


class GSM8KDataset(IDataset):
    @property
    def _name(self) -> DatasetName:
        return DatasetName.GSM8K

    @property
    def _data_cls(self) -> Type[Data]:
        return GSM8KData


class ARCDataset(IDataset):
    @property
    def _name(self) -> DatasetName:
        return DatasetName.ARC

    @property
    def _data_cls(self) -> Type[Data]:
        return ARCData


class LogiQADataset(IDataset):
    @property
    def _name(self) -> DatasetName:
        return DatasetName.LogiQA

    @property
    def _data_cls(self) -> Type[Data]:
        return LogiQAData


Dataset = TypeVar("Dataset", GSM8KDataset, ARCDataset, LogiQADataset)

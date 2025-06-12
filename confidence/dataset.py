import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Type

from pydantic import BaseModel
from typing_extensions import TypeVar

from .data import Data, TimeTablingData, SubsetSumData
from .logger import Record, TimeTablingRecord, SubsetSumRecord


class DatasetName(Enum):
    TimeTabling = "timetabling"
    SubsetSum = "subsetsum"

    def __str__(self) -> str:
        return self.value

    @property
    def record_cls(self) -> Type[Record]:
        record_cls_map = {
            DatasetName.TimeTabling: TimeTablingRecord,
            DatasetName.SubsetSum: SubsetSumRecord,
        }
        return record_cls_map[self]

    @property
    def dataset_cls(self) -> Type["Dataset"]:
        dataset_cls_map = {
            DatasetName.TimeTabling: TimeTablingDataset,
            DatasetName.SubsetSum: SubsetSumDataset,
        }
        return dataset_cls_map[self]


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
        return [data for data in dataset if data.question_id not in already_processed_ids]

    def load_processed_dataset(
        self, already_processed_ids: list[int] | None, force_restart: bool = False
    ) -> list[Data]:
        dataset = self._load_full_dataset()
        if force_restart:
            return []
        return [data for data in dataset if data.question_id in already_processed_ids]


class TimeTablingDataset(IDataset):
    @property
    def _name(self) -> DatasetName:
        return DatasetName.TimeTabling

    @property
    def _data_cls(self) -> Type[Data]:
        return TimeTablingData


class SubsetSumDataset(IDataset):
    @property
    def _name(self) -> DatasetName:
        return DatasetName.SubsetSum

    @property
    def _data_cls(self) -> Type[Data]:
        return SubsetSumData


Dataset = TypeVar("Dataset", TimeTablingDataset, SubsetSumDataset)

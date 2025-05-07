import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Type

from pydantic import BaseModel
from typing_extensions import TypeVar

from .data import Data, GSM8KData, ARCData, LogiQAData, GAOKAOData, TimeTablingData, SubsetSumData
from .logger import GSM8KRecord, ARCRecord, LogiQARecord, GAOKAORecord, Record, TimeTablingRecord, SubsetSumRecord


class DatasetName(Enum):
    GSM8K = "gsm8k"
    ARC = "arc"
    LogiQA = "logiqa"
    GAOKAO_Physics = "gaokao_physics"
    TimeTabling = "timetabling"
    SubsetSum = "subsetsum"

    def __str__(self) -> str:
        return self.value

    @property
    def record_cls(self) -> Type[Record]:
        record_cls_map = {
            DatasetName.GSM8K: GSM8KRecord,
            DatasetName.ARC: ARCRecord,
            DatasetName.LogiQA: LogiQARecord,
            DatasetName.GAOKAO_Physics: GAOKAORecord,
            DatasetName.TimeTabling: TimeTablingRecord,
            DatasetName.SubsetSum: SubsetSumRecord,
        }
        return record_cls_map[self]

    @property
    def dataset_cls(self) -> Type["Dataset"]:
        dataset_cls_map = {
            DatasetName.GSM8K: GSM8KDataset,
            DatasetName.ARC: ARCDataset,
            DatasetName.LogiQA: LogiQADataset,
            DatasetName.GAOKAO_Physics: GAOKAODataset,
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


class GAOKAODataset(IDataset):
    @property
    def _name(self) -> DatasetName:
        return DatasetName.GAOKAO_Physics

    @property
    def _data_cls(self) -> Type[Data]:
        return GAOKAOData


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


Dataset = TypeVar(
    "Dataset",
    GSM8KDataset,
    ARCDataset,
    LogiQADataset,
    GAOKAODataset,
    TimeTablingDataset,
    SubsetSumDataset,
)

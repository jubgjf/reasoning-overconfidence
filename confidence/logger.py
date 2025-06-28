from abc import ABC
from typing import Type, TypeVar

import pandas as pd
from loguru import logger
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from tortoise import Tortoise, fields
from tortoise.contrib.pydantic import pydantic_queryset_creator
from tortoise.models import Model as TortoiseModel

from .data import SubsetSumData, TimeTablingData

TableClass: Type[TortoiseModel]


class IRecord(ABC, BaseModel):
    chat_history: list
    thinking_history: list
    model: str
    dataset: str
    template: str
    temperature: float
    ref: str  # Some addition notes, can be empty
    eval_result: str  # The result of evaluation, can be empty
    git_hash: str


def _make_tabel_cls(record_cls: Type[IRecord], table_name: str):
    table_columns = {}

    for name, field_info in record_cls.model_fields.items():
        name: str
        field_info: FieldInfo
        field_type = field_info.annotation

        if field_type == int:
            if name == "id":
                table_columns[name] = fields.IntField(primary_key=True)
            else:
                table_columns[name] = fields.IntField()
        elif field_type == float:
            table_columns[name] = fields.FloatField()
        elif field_type == str:
            if name == "id":
                table_columns[name] = fields.CharField(max_length=100000, primary_key=True)
            else:
                table_columns[name] = fields.CharField(max_length=100000)
        elif field_type == dict or field_type == dict[str, str] or field_type == list:
            table_columns[name] = fields.JSONField()
        else:
            raise TypeError(f"Unsupported field type {field_type} for {name} ({type(name)})")

    table_columns["Meta"] = type("Meta", (), {"table": table_name})
    global TableClass  # TODO: Bad code, but assign to self._table_cls is not working
    TableClass = type(f"{record_cls.__name__}Table", (TortoiseModel,), table_columns)


class Logger:
    def __init__(
        self,
        db_name: str,
        table_name: str,
        record_cls: Type[IRecord],
        force_update: bool = False,
    ):
        self._db_name = db_name
        self._db_url = f"sqlite://logs/{self._db_name}.db"

        self._record_cls = record_cls
        _make_tabel_cls(self._record_cls, table_name)
        self._orm2pydantic = pydantic_queryset_creator(TableClass)

        self._force_update = force_update

        logger.info(f"Logging url: {self._db_url}::{table_name}")

    async def __aenter__(self):
        await Tortoise.init(db_url=self._db_url, modules={"models": ["confidence.logger"]})
        await Tortoise.generate_schemas()
        if self._force_update:
            logger.warning(f"force_update is True, deleting all records in {TableClass.Meta.table}")
            await TableClass.all().delete()

    async def __aexit__(self, exc_type, exc_value, traceback):
        # Auto clean-up by `tortoise.run_async()`
        ...

    async def insert(self, record: IRecord):
        global TableClass
        await TableClass.update_or_create(**record.model_dump())

    async def fetch(self) -> list[IRecord]:
        global TableClass
        records = await self._orm2pydantic.from_queryset(TableClass.all())
        return [self._record_cls(**r) for r in records.model_dump()]

    async def already_processed_question_ids(self) -> list[int]:
        global TableClass
        records = await TableClass.all().values("question_id")
        return [record["question_id"] for record in records]

    async def history(self) -> dict[str | int, tuple[list[dict[str, str]], list[str]]]:
        global TableClass
        records = await TableClass.all().values("question_id", "chat_history", "thinking_history")
        return {qh["question_id"]: (qh["chat_history"], qh["thinking_history"]) for qh in records}

    async def dump(self) -> pd.DataFrame:
        global TableClass
        records = await self.fetch()
        records = [record.model_dump() for record in records]
        df = pd.DataFrame(records)
        return df


class TimeTablingRecord(IRecord, TimeTablingData): ...


class SubsetSumRecord(IRecord, SubsetSumData): ...


Record = TypeVar("Record", TimeTablingRecord, SubsetSumRecord)

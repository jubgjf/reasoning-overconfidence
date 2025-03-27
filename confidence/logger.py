from abc import ABC
from typing import Type, TypeVar

from loguru import logger
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from tortoise import Tortoise, fields
from tortoise.contrib.pydantic import pydantic_queryset_creator
from tortoise.models import Model as TortoiseModel

from .data import ARCData, GAOKAOData, GSM8KData, LogiQAData

TableClass: Type[TortoiseModel]


class IRecord(ABC, BaseModel):
    model_thinking_response: str
    model_answer_response: str
    model_answer_extracted: str
    model_confidence_response: str
    model_confidence_extracted: float
    template: str
    method: str
    history: dict  # dict[str, str]
    model: str
    ref: str  # Some addition notes, can be empty
    git_hash: str


def list_history_to_dict(history: list[dict[str, str]]) -> dict[str, str]:
    dict_history = {}
    for i, turn in enumerate(history):
        dict_history[f"{turn['role']}_{i // 2}"] = turn["content"]
    return dict_history


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
        elif field_type == dict or field_type == dict[str, str]:
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

    async def chat_history(self) -> dict[str | int, dict[str, str]]:
        global TableClass
        records = await TableClass.all().values("question_id", "history")
        return {qh["question_id"]: qh["history"] for qh in records}


class GSM8KRecord(IRecord, GSM8KData): ...


class ARCRecord(IRecord, ARCData): ...


class LogiQARecord(IRecord, LogiQAData): ...


class GAOKAORecord(IRecord, GAOKAOData): ...


Record = TypeVar("Record", GSM8KRecord, ARCRecord, LogiQARecord, GAOKAORecord)

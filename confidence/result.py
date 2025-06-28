from dataclasses import dataclass


@dataclass
class Result[T, U]:
    ok: T | None = None
    err: U | None = None

    def __post_init__(self):
        if self.ok is None and self.err is None:
            raise ValueError("Either ok_value or err_value must be set.")
        elif self.ok is not None and self.err is not None:
            raise ValueError("Only one of ok_value or err_value can be set at a time.")

    def is_ok(self) -> bool:
        return self.ok is not None

    def is_err(self) -> bool:
        return self.err is not None

    @property
    def ok_value(self) -> T:
        assert self.ok is not None
        return self.ok

    @property
    def err_value(self) -> U:
        assert self.err is not None
        return self.err

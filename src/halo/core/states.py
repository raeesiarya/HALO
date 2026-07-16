from enum import Enum


class DatabaseState(str, Enum):
    FULL = "FULL"
    DEL_ON = "DEL-ON"
    DEL_OFF = "DEL-OFF"


def retrieval_enabled(state: DatabaseState) -> bool:
    return state is not DatabaseState.DEL_OFF

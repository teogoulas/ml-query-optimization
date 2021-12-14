from enum import Enum


class JoinType(Enum):
    NO_JOIN = 0
    HASH_JOIN = 1
    MERGE_JOIN = 2
    LOOP_JOIN = 3

from enum import Enum


class JoinType(Enum):
    NO_JOIN = 0
    HASH_JOIN = 1
    MERGE_JOIN = 2
    LOOP_JOIN = 3

    def getType(self, type: str):
        if type == 'Hash Join':
            return JoinType.HASH_JOIN
        elif type == 'Merge Join':
            return JoinType.MERGE_JOIN
        elif type == 'Nested Loop':
            return JoinType.LOOP_JOIN
        else:
            return JoinType.NO_JOIN

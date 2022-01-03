import numpy as np


class InputNode:
    def __init__(self, is_join: bool, is_scan: bool, is_aggregate: bool, relation: str, left: str, right: str,
                 tables: list):
        self.is_join = is_join
        self.is_scan = is_scan
        self.is_aggregate = is_aggregate
        self.relation = relation if len(relation) > 0 else None
        self.left = left if len(left) > 0 else None
        self.right = right if len(right) > 0 else None
        self.vector = list(np.zeros((3 + len(tables),), dtype=int))
        self.vectorize(tables)

    def vectorize(self, tables: list):
        if self.is_aggregate:
            self.vector[0] = 1
        elif self.is_join:
            self.vector[1] = 1
            self.vector[tables.index(self.left) + 3] = 1
            self.vector[tables.index(self.right) + 3] = 1
        elif self.is_scan:
            self.vector[2] = 1
            self.vector[tables.index(self.left) + 3] = 1

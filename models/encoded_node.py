import array

import numpy as np

from models.join_type import JoinType
from models.scan_type import ScanType


class EncodedNode:
    def __init__(self, join_types: int, relations_number: int, joinType: JoinType, scanType: ScanType,
                 own_relations: array, left_relations: array, right_relations: array):
        self.left = None
        self.right = None
        self.join_types = join_types
        self.relations_number = relations_number
        self.vector = np.zeros((join_types + 2 * relations_number,), dtype=int)
        self.type = joinType if joinType != JoinType.NO_JOIN else scanType if scanType != ScanType.NO_SCAN else None
        self.insert(joinType, scanType, own_relations, left_relations, right_relations)

    def insert(self, joinType: JoinType, scanType: ScanType, relations: array, left_relations: array,
               right_relations: array):
        if scanType != ScanType.NO_SCAN:
            try:
                self.vector[self.join_types - 1 + 2 * relations.index(1) + scanType.value - 1] = 1
            except:
                print('No relation!')
        elif joinType != JoinType.NO_JOIN:
            self.vector[joinType.value - 1] = 1
            for i in range(self.join_types, len(self.vector) + 1):
                self.vector[i] = (left_relations[i] + right_relations[i]) % 2

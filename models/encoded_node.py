import array

import numpy as np

from models.join_type import JoinType
from models.scan_type import ScanType


class EncodedNode:
    def __init__(self, join_types: int, relations_number: int, joinType: JoinType, scanType: ScanType,
                 own_relations: list, left_node, right_node):
        self.left = left_node
        self.right = right_node
        self.join_types = join_types
        self.relations_number = relations_number
        self.vector = list(np.zeros((join_types + 2 * relations_number,), dtype=int))
        self.type = joinType if joinType != JoinType.NO_JOIN else scanType if scanType != ScanType.NO_SCAN else None
        self.insert(joinType, scanType, own_relations,
                    left_node.vector[-2 * relations_number:] if left_node is not None else None,
                    right_node.vector[-2 * relations_number:] if right_node is not None else None)

    def insert(self, joinType: JoinType, scanType: ScanType, relations: list, left_relations: list,
               right_relations: list):
        if scanType != ScanType.NO_SCAN:
            try:
                self.vector[self.join_types - 1 + 2 * relations.index(1) + scanType.value - 1] = 1
            except:
                print('No relation!')
        elif joinType != JoinType.NO_JOIN:
            self.vector[joinType.value - 1] = 1
            for i in range(self.join_types, len(self.vector)):
                self.vector[i] = (left_relations[i - self.join_types] + right_relations[i - self.join_types]) % 2

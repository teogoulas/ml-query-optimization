import array

import numpy as np

from models.join_type import JoinType
from models.scan_type import ScanType
from utils.constants import JOIN_NUMBER, SCAN_NUMBER


class EncodedNode:
    def __init__(self, relations_number: int, joinType: JoinType, scanType: ScanType,
                 own_relations: list, left_node, right_node):
        self.left = left_node
        self.right = right_node
        self.relations_number = relations_number
        self.vector = list(np.zeros((JOIN_NUMBER + SCAN_NUMBER * relations_number,), dtype=int))
        self.type = joinType if joinType != JoinType.NO_JOIN else scanType if scanType != ScanType.NO_SCAN else None
        self.insert(joinType, scanType, own_relations,
                    left_node.vector[-SCAN_NUMBER * relations_number:] if left_node is not None else None,
                    right_node.vector[-SCAN_NUMBER * relations_number:] if right_node is not None
                    else np.zeros(SCAN_NUMBER * relations_number, dtype=int) if joinType is JoinType.AGGREGATE else None)

    def insert(self, joinType: JoinType, scanType: ScanType, relations: list, left_relations: list,
               right_relations: list):
        if scanType != ScanType.NO_SCAN:
            try:
                self.vector[JOIN_NUMBER - 1 + SCAN_NUMBER * relations.index(1) + scanType.value - 1] = 1
            except:
                print('No relation!')
        elif joinType != JoinType.NO_JOIN:
            self.vector[joinType.value - 1] = 1
            for i in range(JOIN_NUMBER, len(self.vector)):
                self.vector[i] = (left_relations[i - JOIN_NUMBER] + right_relations[i - JOIN_NUMBER]) % SCAN_NUMBER

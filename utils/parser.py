import importlib

import numpy as np
from six import next, iterkeys, itervalues

from models.database import Database
from models.encoded_node import EncodedNode
from models.join_type import JoinType
from models.scan_type import ScanType

module = importlib.import_module('utils')


def get_join_type(join_type: str):
    if join_type == 'Hash Join':
        return JoinType.HASH_JOIN
    elif join_type == 'Merge Join':
        return JoinType.MERGE_JOIN
    elif join_type == 'Nested Loop':
        return JoinType.LOOP_JOIN
    else:
        return JoinType.NO_JOIN


def get_scan_type(scan_type: str):
    scan_type = scan_type.lower()
    if 'index' in scan_type:
        return ScanType.INDEX
    elif 'scan' in scan_type:
        return ScanType.TABLE
    else:
        return ScanType.NO_SCAN


def create_scan_node(json, tables, child):
    relations = np.zeros(len(tables), dtype=int)
    scan_type = get_scan_type(json['Node Type'])
    try:
        if 'Relation Name' in json.keys():
            table_name = json['Relation Name']
        else:
            index_name = json['Index Cond'][1:json['Index Cond'].index(" =")]
            table_name = json['Index Name'].replace(index_name + "_", "")
        relation_index = tables.index(table_name)
        relations[relation_index] = 1
    except:
        print("No table found for node type {}".format(json['Node Type']))
    return EncodedNode(len(JoinType) - 1, len(tables), JoinType.NO_JOIN, scan_type, list(relations), child, None)


def parse_json(json, db: Database):
    if 'Plans' in json.keys():
        children = []
        for plan in json['Plans']:
            children.append(parse_json(plan, db))
        if len(children) > 1:
            return EncodedNode(len(JoinType) - 1, len(db.tables), get_join_type(json['Node Type']), ScanType.NO_SCAN,
                               [], children[0], children[1])
        else:
            return create_scan_node(json, db.tables, children[0])
    else:
        return create_scan_node(json, db.tables, None)


def flatten_tree(root: EncodedNode):
    feature_vector = root.vector
    if root.left is None and root.right is None:
        return feature_vector
    else:
        if root.left is not None:
            feature_vector = feature_vector + flatten_tree(root.left)
        if root.right is not None:
            feature_vector = feature_vector + flatten_tree(root.right)
        return feature_vector

import importlib

import numpy as np
import sqlparse
from typing import Iterator, Tuple
import re

from sqlparse.sql import Statement, TokenList, Identifier, Token, IdentifierList, Where, Comparison, Parenthesis

from models.database import Database
from models.encoded_node import EncodedNode
from models.join_type import JoinType
from models.json_node import InputNode
from models.scan_type import ScanType
from utils.constants import AGGREGATE_FN_LIST

module = importlib.import_module('utils')

JOIN_COUNTER = 2


def get_join_type(join_type: str):
    if join_type == 'Hash Join':
        return JoinType.HASH_JOIN
    elif join_type == 'Merge Join':
        return JoinType.MERGE_JOIN
    elif join_type == 'Nested Loop':
        return JoinType.LOOP_JOIN
    elif join_type == 'Aggregate':
        return JoinType.AGGREGATE
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
    return EncodedNode(len(tables), JoinType.NO_JOIN, scan_type, list(relations), child, None)


def parse_json(json, db: Database):
    if 'Plans' in json.keys():
        children = []
        for plan in json['Plans']:
            children.append(parse_json(plan, db))
        if len(children) > 1:
            return EncodedNode(len(db.tables), get_join_type(json['Node Type']), ScanType.NO_SCAN,
                               [], children[0], children[1])
        elif len(children) == 1 and get_join_type(json['Node Type']) == JoinType.AGGREGATE:
            return EncodedNode(len(db.tables), JoinType.AGGREGATE, ScanType.NO_SCAN,
                               [], children[0], None)
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


def parsed_statement_and_coords(text: str) -> Iterator[Tuple[Statement, int]]:
    """
    Parse a text contains SQL statements.
    :param text: The text contains SQL statements.
    :return: Iterator yields tuples of parsed statement and its line number.
    """
    # Remove the last separator to avoid to get empty statement.
    text = text.rstrip(';\n')

    line_index = 0
    sql_end_index = 0
    for sql in sqlparse.split(text):
        sql_start_index = sql_end_index
        # Determine the coordinates of the statement.
        sql_end_index = text.find(sql, sql_start_index)
        assert sql_end_index != -1

        # Update the current line number.
        line_index += text.count('\n', sql_start_index, sql_end_index)

        # Remove semicolon at the end.
        sql = sql.rstrip(';')

        # Yield the parsed statement and line number.
        yield sqlparse.parse(sql)[0], line_index + 1


def repr_token(token: sqlparse.sql.Token):
    """
    Return a string that represents a SQL token.
    :param token: The token will be explained.
    :return: A single-line string that represents the token.
    """
    if isinstance(token, TokenList):
        # Show class name if the token is instance of TokenList.
        typename = type(token).__name__
    else:
        # As for Token, show the token type from ttype field.
        typename = 'Token(ttype={0})'.format(str(token.ttype).split('.')[-1])

    value = str(token)
    if len(value) > 30:
        value = value[:29] + '...'
    value = re.sub(r'\s+', ' ', value)
    q = '"' if value.startswith("'") and value.endswith("'") else "'"

    details = {}
    if isinstance(token, TokenList):
        details['alias'] = token.get_alias()
        details['name'] = token.get_name()
        details['parent_name'] = token.get_parent_name()
        details['real_name'] = token.get_real_name()
    if isinstance(token, Identifier):
        details['ordering'] = token.get_ordering()
        details['typecast'] = token.get_typecast()
        details['is_wildcard'] = token.is_wildcard()

    return '{type} {q}{value}{q} {detail}'.format(type=typename, q=q, value=value,
                                                  detail=repr(details) if details else '')


def print_tree(tokens: sqlparse.sql.TokenList, left=''):
    """
    Print SQL tokens as a tree.
    :param tokens: TokenList object to be printed.
    :param left: Left string printed for each lines. (normally it's for internal use)
    """
    num_tokens = len(tokens)
    for i, token in enumerate(tokens):
        last = i + 1 == num_tokens
        horizontal_node = '├' if not last else '└'
        vertical_node = '│' if not last else '  '

        print('{left}{repr}'.format(left=left + horizontal_node, repr=repr_token(token)))
        if isinstance(token, TokenList):
            if token.is_group:
                print_tree(token.tokens, left=left + vertical_node)


def handle_comparison(token: Token, tables_array: np.array, columns_array: np.array, steps: list, tables: list,
                      aliases: dict, column_array_index: list) -> (np.array, np.array, list):
    for t in token.tokens:
        global JOIN_COUNTER
        if isinstance(t, Comparison):
            if isinstance(t.left, Identifier) and isinstance(t.right, Identifier):
                steps.append(InputNode(True, False, False, t.value, aliases[t.left.value.split('.')[0]],
                                       aliases[t.right.value.split('.')[0]], tables))
                tables_array[tables.index(aliases[t.left.value.split('.')[0]])][
                    tables.index(aliases[t.right.value.split('.')[0]])] = 1
                tables_array[tables.index(aliases[t.right.value.split('.')[0]])][
                    tables.index(aliases[t.left.value.split('.')[0]])] = 1
                columns_array[column_array_index.index(
                    aliases[t.left.value.split('.')[0]] + "_" + t.left.value.split('.')[1])] = JOIN_COUNTER
                columns_array[column_array_index.index(
                    aliases[t.right.value.split('.')[0]] + "_" + t.right.value.split('.')[1])] = JOIN_COUNTER
                JOIN_COUNTER += 1
            elif isinstance(t.left, Identifier) and isinstance(t.right, Token):
                steps.append(InputNode(False, True, False, t.value, aliases[t.left.value.split('.')[0]], '', tables))
                tables_array[tables.index(aliases[t.left.value.split('.')[0]])][
                    tables.index(aliases[t.left.value.split('.')[0]])] = 1
                columns_array[column_array_index.index(
                    aliases[t.left.value.split('.')[0]] + "_" + t.left.value.split('.')[1])] = 1
        elif isinstance(t, Parenthesis):
            tables_array, columns_array, steps =\
                handle_comparison(t, tables_array, columns_array, steps, tables, aliases, column_array_index)
    return tables_array, columns_array, steps


def parse_sql_statement(tokens: sqlparse.sql.TokenList, tables: list, column_array_index: list) -> list:
    aliases = {}
    steps = []
    select_statement = False
    from_statement = False
    tables_array = np.zeros(shape=(len(tables), len(tables)), dtype=int)
    columns_array = np.zeros(len(column_array_index), dtype=int)
    for token in tokens:
        if isinstance(token, IdentifierList):
            if select_statement:
                if any(fn in token.value.lower() for fn in AGGREGATE_FN_LIST):
                    steps.append(InputNode(False, False, True, token.value, '', '', tables))
            elif from_statement:
                for t in token.tokens:
                    if isinstance(t, Identifier) and ' as ' in t.value.lower():
                        al = t.value.lower().split(' as ')
                        aliases[al[1].strip()] = al[0].strip()
        elif isinstance(token, Where):
            from_statement = False
            tables_array, columns_array, steps =\
                handle_comparison(token, tables_array, columns_array, steps, tables, aliases, column_array_index)
        elif isinstance(token, Token):
            if token.value.lower() == "select":
                select_statement = True
            elif token.value.lower() == "from":
                select_statement = False
                from_statement = True
            continue

    return list(np.concatenate([tables_array[np.triu_indices(len(tables))], columns_array]))

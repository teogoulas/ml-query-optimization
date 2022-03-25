import importlib
import re
from typing import Iterator, Tuple

import numpy as np
import sqlparse
from sqlparse.sql import Statement, TokenList, Identifier, Token, IdentifierList, Where, Comparison, Parenthesis

from models.database import Database
from models.encoded_node import EncodedNode
from models.join_type import JoinType
from models.json_node import InputNode
from models.scan_type import ScanType
from utils.constants import AGGREGATE_FN_LIST

module = importlib.import_module('utils')

JOIN_COUNTER = 2


def generate_input_text(predicates: dict, aliases: dict) -> str:
    input_text = ''
    if len(predicates) == 0:
        for alias, table in enumerate(aliases):
            input_text += 'scan {} '.format(table)
    else:
        for (pt, pv) in predicates:
            lh, lha, rh, rha = pv
            if pt == 'scan':
                input_text += "{0} {1}.{2} " \
                    .format(pt, aliases[lh], lha)
            elif pt == 'join':
                input_text += "{0} {1}.{2}-{3}.{4} " \
                    .format(pt, aliases[lh], lha, aliases[rh], rha)
    return input_text[:-1]


def generate_output_text(json: dict, aliases: dict) -> str:
    output_string = ''
    if 'Plans' in json.keys():
        output_string += (str(json['Join Type']).lower() + ' ' if 'Join Type' in json.keys() else '') + str(
            json['Node Type']).lower() + ' '
        if 'Hash Cond' in json.keys():
            hash_cond = json['Hash Cond'].replace(" ", "").replace('(', '').replace(')', '')
            if '=' in hash_cond:
                preds = hash_cond.split("=")
                for n, pred in enumerate(preds):
                    if '.' in pred:
                        table, column = pred.split('.')
                        if table in aliases.keys():
                            table = aliases[table]
                        output_string += '{0}.{1}'.format(table, column) + ('-' if len(preds) > 1 and n == 0 else ' ')
                    else:
                        output_string += pred + ('-' if len(preds) > 1 and n == 0 else ' ')

        for plan in json['Plans']:
            output_string += generate_output_text(plan, aliases)
    else:
        if 'Index Cond' in json.keys():
            output_string += str(json['Node Type']).lower()
            if 'Relation Name' in json.keys():
                output_string += " {0}".format(str(json['Relation Name']).lower())
            index_cond = json['Index Cond']
            if len(index_cond.split()) > 0:
                output_string += ".{0}".format(str(json['Index Cond'].split()[0].replace("(", "")).lower())
                predicates = index_cond.split("=")
                if len(predicates) > 1:
                    second_predicate = predicates[1].replace(" ", "").replace(")", "")
                    cmp = second_predicate.split(".")
                    if len(cmp) > 1 and aliases[cmp[0]] is not None:
                        output_string += "-{0}.{1}".format(str(aliases[cmp[0]]).lower(), str(cmp[1]).lower())

            output_string += " "
        elif 'Filter' in json.keys():
            output_string += str(json['Node Type']).lower()
            statement = re.sub("[^_0-9a-zA-Z]+", " ", json['Filter']).split()
            if len(statement) > 0:
                output_string += " table scan {0}.{1}".format(json['Relation Name'].lower(), statement[0])

            output_string += " "

        elif 'scan' in str(json['Node Type']).lower():
            output_string += " {0} {1} ".format(str(json['Node Type']).lower(), str(json['Relation Name'].lower()))

    return output_string


def generate_operation_text(json: dict, operators: dict, aliases: dict, simplyfied=False) -> str:
    output_string = ''

    if len(operators) > 0:
        for (operator, predicates) in operators:
            results = simplified_parse_plans(json, operator, predicates) if simplyfied else parse_plans(json, operator, predicates, aliases)
            output_string += results[0] + ' '
    else:
        # alias = ''
        # for key in json.keys():
        #     if json[key] in aliases.keys():
        #         alias = aliases[json[key]]
        #         break
        output_string = (simplified_parse_simple_plans(json, aliases)[0] if simplyfied else parse_simple_plans(json, aliases)) + ' '

    return output_string


def parse_simple_plans(json: dict, aliases: dict) -> str:
    output_string = ''
    node_type = str(json['Node Type']).lower().replace(' ', '_')
    if 'Plans' in json.keys():
        for plan in json['Plans']:
            out = parse_simple_plans(plan, aliases)
            output_string = out + ('-' if len(out) > 0 and len(output_string) > 0 else '') + output_string
    else:
        if 'hash' in node_type or 'scan' in node_type:
            # alias = ''
            # for key in json.keys():
            #     if json[key] in aliases.keys():
            #         alias = aliases[json[key]]
            #         break
            output_string = node_type  # + (f"~{alias}" if len(alias) > 0 else '')

    return output_string


def parse_plans(json: dict, operator: str, predicates: list, aliases: dict) -> Tuple[str, bool, bool]:
    output_string = ''
    force_stop = False
    g_found = False
    node_type = str(json['Node Type']).lower().replace(' ', '_')
    if 'Plans' in json.keys():
        dirty_output = ''
        for plan in json['Plans']:
            output, found, terminate = parse_plans(plan, operator, predicates, aliases)
            if terminate:
                force_stop = True
                output_string = (
                                    node_type + '-'
                                    if operator == 'scan'
                                       and ('hash' in node_type or 'scan' in node_type)
                                       and 'Join Type' not in json.keys()
                                    else ''
                                ) + output
            if found:
                g_found = True
                if not terminate:
                    if operator == 'join' and 'Join Type' not in json.keys():
                        force_stop = True
                        output_string = output
                    elif operator == 'scan' and 'Join Type' in json.keys():
                        force_stop = True
                        output_string = output + (
                            '-' if len(output) > 0 and len(dirty_output) > 0 else '') + dirty_output
            else:
                if not g_found:
                    for key in json.keys():
                        if predicates[1] in str(json[key]) and predicates[3] in str(json[key]) and key != 'Plans':
                            g_found = True
                            break
                dirty_output = output + ('-' if len(output) > 0 and len(dirty_output) > 0 else '') + dirty_output
        if force_stop:
            return output_string, g_found, force_stop
        else:
            if g_found:
                output_string = (dirty_output + '-' if len(dirty_output) > 0 else '') + node_type
            else:
                output_string = node_type + ('-' + dirty_output if len(dirty_output) > 0 else '')
            return output_string, g_found, force_stop

    else:
        alias = ''
        for key in json.keys():
            if predicates[1] in str(json[key]) and predicates[3] in str(json[key]):
                g_found = True
            if json[key] in aliases.keys():
                alias = aliases[json[key]]
        output_string = str(json['Node Type']).lower().replace(' ', '_')  # + (f"~{alias}" if len(alias) > 0 else '')
        return output_string, g_found, g_found and operator == 'scan'


def simplified_parse_plans(json: dict, operator: str, predicates: list) -> Tuple[str, bool]:
    output_string = ''
    node_type = str(json['Node Type']).lower().replace(' ', '_')
    found = False
    if 'Plans' in json.keys():
        for plan in json['Plans']:
            output, found = simplified_parse_plans(plan, operator, predicates)
            if found:
                output_string = output
                break
        if not found:
            for key in json.keys():
                if key != 'Plans' and predicates[1] in str(json[key]) and predicates[3] in str(json[key]):
                    output_string = node_type
                    found = True
                    break
        return output_string, found
    else:
        for key in json.keys():
            if predicates[1] in str(json[key]) and predicates[3] in str(json[key]):
                found = True
                break
        return node_type, found


def simplified_parse_simple_plans(json: dict, aliases: dict) -> Tuple[str, bool]:
    output_string = ''
    node_type = str(json['Node Type']).lower().replace(' ', '_')
    found = False
    if 'Plans' in json.keys():
        for plan in json['Plans']:
            out, found = simplified_parse_simple_plans(plan, aliases)
            if found:
                output_string = out
                break
        return output_string, found
    else:
        if 'hash' in node_type or 'scan' in node_type:
            found = True
        return node_type, found


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
            tables_array, columns_array, steps = \
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
            tables_array, columns_array, steps = \
                handle_comparison(token, tables_array, columns_array, steps, tables, aliases, column_array_index)
        elif isinstance(token, Token):
            if token.value.lower() == "select":
                select_statement = True
            elif token.value.lower() == "from":
                select_statement = False
                from_statement = True
            continue

    return list(np.concatenate([tables_array[np.triu_indices(len(tables))], columns_array]))

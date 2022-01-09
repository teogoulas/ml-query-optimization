import os

import numpy as np
import sqlparse

from classification.classifier import svm_classification
from models.database import Database
from utils.parser import parse_json, flatten_tree, parse_sql_statement


def main():
    dataset = "data/queries"
    cwd = os.getcwd()
    files = os.listdir(os.path.join(cwd, *dataset.split("/")))

    db = Database(collect_db_info=True)
    column_array_index = []
    for table, columns in db.tables_attributes.items():
        for column in columns:
            column_array_index.append(table + "_" + column)

    X = []
    y = []

    y_length_max = 0
    for file in files:
        f = open(dataset + "/" + file, "r")
        query = f.read()

        rows = db.explain_query(query)
        statement = sqlparse.parse(query)[0]
        x = parse_sql_statement(statement.tokens, db.tables, column_array_index)
        X.append(x)
        encoded_node = parse_json(rows, db)
        y_l = flatten_tree(encoded_node)
        if y_length_max < len(y_l):
            y_length_max = len(y_l)
        y.append(y_l)

    Y = []
    for y_l in y:
        Y.append((y_l + y_length_max * [0])[:y_length_max])

    svm_classification(np.array(X), np.array(Y))


if __name__ == "__main__":
    main()

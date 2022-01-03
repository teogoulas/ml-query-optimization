import os

from models.database import Database
from utils.parser import parse_json, flatten_tree, parsed_statement_and_coords, print_tree, parse_sql_statement


def main():
    dataset = "data/queries"
    cwd = os.getcwd()
    files = os.listdir(os.path.join(cwd, *dataset.split("/")))

    db = Database(collect_db_info=True)
    ## for file in files:
    ## f = open(dataset + "/" + files[0], "r")
    f = open(dataset + "/1a.sql", "r")
    query = f.read()

    rows = db.explain_query(query)
    for statement, line in parsed_statement_and_coords(query):
        print('{0}:{1}'.format(os.path.relpath(files[0]), line))
        print_tree(statement.tokens)
        root = parse_sql_statement(statement.tokens, db.tables)
    encoded_node = parse_json(rows, db)
    print(flatten_tree(encoded_node))
    print("almost")


if __name__ == "__main__":
    main()

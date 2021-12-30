import os
import psqlparse

from models.database import Database
from utils.parser import parse_json, flatten_tree


def main():
    dataset = "data/queries"
    cwd = os.getcwd()
    files = os.listdir(os.path.join(cwd, *dataset.split("/")))

    db = Database(collect_db_info=True)
    for file in files:
        f = open(dataset + "/" + file, "r")
        query = f.read()

        statements = psqlparse.parse('SELECT * from mytable')
        used_tables = statements[0].tables()
        rows = db.explain_query(query)
        encoded_node = parse_json(rows, db)
        print(flatten_tree(encoded_node))
    print("almost")


if __name__ == "__main__":
    main()

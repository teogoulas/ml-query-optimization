import os

from models.database import Database
from utils.parser import build_from_obj


def main():
    dataset = "data/queries"
    cwd = os.getcwd()
    files = os.listdir(os.path.join(cwd, *dataset.split("/")))

    db = Database(collect_db_info=False)
    cursor = db.conn.cursor()
    file = open(dataset + "/" + files[0], "r")
    query = file.read()
    rows = db.explain_query(query)
    parsed = build_from_obj(rows[0][0][0])


if __name__ == "__main__":
    main()

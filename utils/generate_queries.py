import json


def main():
    # Opening JSON file
    with open('../data/queries_backup/cosql_train.json') as in_file:
        data = json.load(in_file)

        with open('../data/queries/queries_train.sql', 'w+', encoding='utf-8') as out_file:
            for el in data:
                out_file.write(str(el["query"].strip()).lower().replace('"', "'") + ";\n")


if __name__ == "__main__":
    main()

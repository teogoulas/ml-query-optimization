import os

from models.database import Database
from models.job_query import JOBQuery
from utils.parser import generate_output_text, generate_input_text
from utils.vectorization import bag_of_characters


def main():
    dataset = "data/queries"
    cwd = os.getcwd()
    files = os.listdir(os.path.join(cwd, *dataset.split("/")))

    db = Database(collect_db_info=True)
    column_array_index = []
    for table, columns in db.tables_attributes.items():
        for column in columns:
            column_array_index.append(table + "_" + column)

    # initialize all variables
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()

    for file in files:
        f = open(dataset + "/" + file, "r")
        query = f.read().strip()
        job_query = JOBQuery(query)
        rows = db.explain_query(query)

        input_text = generate_input_text(job_query.predicates, job_query.rel_lookup)
        input_texts.append(input_text)
        # add '\t' at start and '\n' at end of text.
        target_text = '\t' + generate_output_text(rows, job_query.rel_lookup)[:-2] + '\n'
        target_texts.append(target_text)

        # split character from text and add in respective sets
        input_characters.update(list(input_text.lower()))
        target_characters.update(list(target_text.lower()))

    # sort input and target characters
    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    # get the total length of input and target characters
    num_en_chars = len(input_characters)
    num_dec_chars = len(target_characters)
    # get the maximum length of input and target text.
    max_input_length = max([len(i) for i in input_texts])
    max_target_length = max([len(i) for i in target_texts])
    print("number of encoder characters : ", num_en_chars)
    print("number of decoder characters : ", num_dec_chars)
    print("maximum input length : ", max_input_length)
    print("maximum target length : ", max_target_length)

    en_in_data, dec_in_data, dec_tr_data = bag_of_characters(input_texts, target_texts, input_characters, target_characters)
    print("number of encoder input data : ", len(en_in_data))
    print("number of decoder input data : ", len(dec_in_data))
    print("number of decoder training data : ", len(dec_tr_data))


if __name__ == "__main__":
    main()

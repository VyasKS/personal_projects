import question_answering

# Run long_document_search first when starting with a fresh data. Json file should contain a field 'text', else refactor the code suiting the data format used. This creates an index and
# saves to data folder which will be later fed to downstream modeling tasks.


def main(question, k):
    o = question_answering.QuestionAnswering(query=question, k=k)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(question="how many people died from black death?", k=3)


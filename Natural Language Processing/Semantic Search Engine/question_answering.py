import json
import torch
import faiss
from transformers import BertTokenizer, BertForQuestionAnswering
from sentence_transformers import SentenceTransformer
from FAISS_indexer import Semantic


class QuestionAnswering(Semantic):

    """ param: Query - User query passed as a question
        param: k - list of top answers to show as output
        returns: list of contextually relevant answers for the user query"""

    def __init__(self, query, k=5):
        """ Constructor """
        self.documents = self.get_data()
        self.query = query
        self.k = k
        self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        context = self.get_context()
        answers = self.get_answers()
        print(f"User asked for {self.k} top results.\n"
              f"Most precise answer for the query is {answers[0]}\n"
              f"Top answers for the query are : {answers[:-1]}\n"   # Ignores final [SEP] token
              f"Search results and their scores are {context}")

    @classmethod
    def get_data(cls):
        with open('data/data.json', 'r') as f:
            documents = json.load(f)
            corpus = [d['text'] for d in documents]
        return corpus

    def build_index(self):
        """ Overrides the method in parent class """
        index = faiss.read_index('data/pandemics')
        return index

    def get_context(self):
        embed = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        encoded_query = embed.encode([self.query])
        index = self.build_index()
        top_k = index.search(encoded_query, self.k)
        corpus = self.documents
        answers = [corpus[_id] for _id in top_k[1].tolist()[0]]
        scores = top_k[0][0]
        return list(zip(answers, scores))

    def get_answers(self):
        bert_model = self.model
        bert_tokenizer = self.tokenizer
        # Perform tokenization of input text
        reference_text = self.get_context() # A list of contextual passages upto a number passed in at 'k'. This is where BERT will look for an answer.
        list_of_answers = []
        for context, score in reference_text:
            try:
                input_ids = bert_tokenizer.encode(self.query, context)
                input_tokens = bert_tokenizer.convert_ids_to_tokens(input_ids)
                # Find index of first occurrence of [SEP] token
                sep_location = input_ids.index(bert_tokenizer.sep_token_id)
                first_segment_length, second_segment_length = (sep_location + 1, len(input_ids) - (sep_location + 1))
                segment_embedding = [0] * first_segment_length + [1] * second_segment_length
                # Test model on query
                model_scores = bert_model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_embedding]))
                answer_start_location, answer_end_location = torch.argmax(model_scores[0]), torch.argmax(model_scores[1])
                result = ' '.join(input_tokens[answer_start_location:answer_end_location + 1])
                result = result.replace(' ##', '')
                list_of_answers.append(result)
            except IOError:
                print("Error occurred in getting answers")
        return list_of_answers



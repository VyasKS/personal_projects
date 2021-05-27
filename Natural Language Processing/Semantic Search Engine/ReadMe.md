In this project we will use transfer learning with BERT to build a semantic search engine.
BERT is a type of neural network model that revolutionized Natural Language Processing because of the advantage it presents in learning context. Because it is bidirectional, BERT can understand the meaning of each word based on context both to the right and to the left of it. BERT excels in many traditional NLP tasks, like search, summarization and question answering. To search documents, we need to frame the problem as a text similarity task. We can apply BERT to computer vector representations of documents. Unlike statistical methods relying on counting and matching words in the query and the documents, BERT allows to find similar documents based on their contextually meaningful vector representations. You will apply several different libraries and models to build a search engine with BERT:

    BERT model to vectorize the documents. Faiss library to create an inverted index for indexing documents in their vectorized form and similarity search.
    Sentence-transformers library to encode and search longer documents. Sentence-transformers is a framework that provides pre-trained Transformers models,
    specifically trained to create meaningful embeddings for sentences and paragraphs. Pre-trained BERT models fine-tuned for factual question answering
    tasks to extract a concise answer to your search query. Flask API for deploying the model on a server in development where HTTP request can be sent for
    retrieving API response.

Project outline:
1. Semantic Search Engine with 'Faiss' and 'BERT'

2. Searching long documents with Sentence Transformers (indexes sentences)

3. Question Answering with BERT

Libraries used:
Please install these libraries for execution (no requirements.txt necessary, hence not provided)

    Json, Faiss, Numpy, PyTorch, Transformers, Sentence-Transformers, Flask-restful

Order of execution:
1. long_document_search.py for initial indexing of the data. Data as a json should contain the field
'text', and the relevant string value. Else, refactor the code that suits dataset used.
   
2. main.py executes question_answering.py that inherits features from Faiss_indexer.py

3. Deploy.py script will host the model on a server and can be listened with the port
detail mentioned in the script. The code processes input passed to an API, calls the function qa_system(),
   and sends a response from this function as an API response.
   
Postman was used to test the code and all parameters were working correctly. Please create a request body
inside the query parameter with parameters 'question' where the actual question is passed and value of 'k'
displays top 'k' search results, and their respective context passages and scores as well, default is set to 3.
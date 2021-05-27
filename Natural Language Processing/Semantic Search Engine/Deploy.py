""" This script runs and deploys the whole search engine with question answering capability for the consumption of a website or a conversation system to serve the end user
who is looking for an answer to his or her query. It needs to release or expose features of the QA system as a REST API. To generate a public URL inside a private network use 'ngrok'
for QA system on both Windows or Linux Server. A RESTful API uses HTTP requests to GET and POST data. """
from flask import Flask, request
import json
from question_answering import QuestionAnswering
app = Flask(__name__)


# this is the URI if served on a web server. For now, a method is sufficient for testing.
@app.route("/qa_system", methods=['POST'])
def qa_system():
    try:
        # sets query parameters similar to postman service
        json_data = request.get_json(force=True)
        query = json_data['query']
        context_list = json_data['context_list']
        result = []
        for value in context_list:
            context = value['context']
            context = context.replace("\n", " ")
            answer_json_final = dict()
            answer = QuestionAnswering(query=query)
            answer_json_final['answer'] = answer
            answer_json_final['id'] = value['id']
            answer_json_final['question'] = query
            result.append(answer_json_final)

        result = {'results': result}
        result = json.dumps(result)
        return result
    except Exception as e:
        return {"Error": str(e)}


if __name__ == "__main__":
    app.run(debug=True, port='5000')

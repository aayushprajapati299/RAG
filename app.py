"""
app.py

Hey! This is the main file that kicks off our RAG web app. 
Basically, it sets up the Flask server, dishes out the frontend, and handles the API routes. 
We've got an upload route for dumping text/PDF files into the index, and an ask route to actually query the stuff we indexed.
"""

import os
from flask import Flask, request, jsonify, render_template  
from werkzeug.utils import secure_filename

# Pull in our custom stuff
from indexer import index_document
from retriever import retrieve_pages
from generator import generate_answer

app = Flask(__name__)

# Setup where we're gonna drop the uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def index():
    """Just sends over the main page so the user has something to look at."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Grabs an uploaded PDF or text file and throws it over to the indexer.
    Let's the frontend know how many pages we ended up indexing.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Hand it off to the indexer to do the heavy lifting
            pages_indexed = index_document(filepath)
            return jsonify({
                "status": "ok", 
                "pages": pages_indexed
            }), 200
            
        except Exception as e:
            return jsonify({"error": f"Failed to index document: {str(e)}"}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Takes a question from the frontend, digs up some relevant pages, 
    and gets the LLM to spit out an answer. Returns the answer and where we found it.
    """
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' field in the request body"}), 400
        
    question = data['question']
    
    try:
        pages = retrieve_pages(question)
        result = generate_answer(question, pages)
        return jsonify({
            "answer": result["answer"],
            "sources": result["sources"]
        }), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to generate answer: {str(e)}"}), 500

if __name__ == "__main__":
    # Fire up the server! Listening on all interfaces so we can reach it easily.
    app.run(host="0.0.0.0", port=5000, debug=True)

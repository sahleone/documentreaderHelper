import nltk
from nltk.tokenize import sent_tokenize
import ssl

# Add this function at the beginning of your file
def download_nltk_data():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    nltk.download('punkt')

# Call this function before you use any NLTK functionality
download_nltk_data()


from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_from_directory
from datetime import timedelta  # Added this import
from werkzeug.utils import secure_filename
import os
import fitz  # PyMuPDF for PDF handling
from docx import Document  # python-docx for DOCX handling
from epub_conversion.utils import open_book, convert_epub_to_lines
import html2text  # Add this import
import secrets  # Import secrets module
import tempfile
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import re
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import PyPDF2


app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Generates a random 32-character hex string

# Define the upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Load environment variables
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Device selection
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f"Using device: {device}")

# Initialize sentence transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Load the model and tokenizer for question answering
model_name = "distilbert-base-cased-distilled-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a question-answering pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)

# Initialize Faiss index
dimension = 384  # Dimension of the sentence embeddings
faiss_index = faiss.IndexFlatL2(dimension)

# Global variable to store document contents
documents = {}

def preprocess_and_index_document(content, doc_id):
    sentences = sent_tokenize(content)
    embeddings = sentence_model.encode(sentences)
    
    # Check if embeddings is a PyTorch tensor
    if torch.is_tensor(embeddings):
        # Move embeddings to CPU if they're on GPU or MPS
        if device != 'cpu':
            embeddings = embeddings.cpu().numpy()
        else:
            embeddings = embeddings.numpy()
    else:
        # If it's already a NumPy array, no need to convert
        embeddings = np.array(embeddings)
    
    # Add embeddings to Faiss index
    faiss.normalize_L2(embeddings)
    faiss_index.add(embeddings)
    
    # Store sentences with their document ID
    for i, sentence in enumerate(sentences):
        documents[faiss_index.ntotal - len(sentences) + i] = (doc_id, sentence)

def retrieve_relevant_context(query, top_k=5):
    query_vector = sentence_model.encode([query])
    
    # Check if query_vector is a PyTorch tensor
    if torch.is_tensor(query_vector):
        # Move query_vector to CPU if it's on GPU or MPS
        if device != 'cpu':
            query_vector = query_vector.cpu().numpy()
        else:
            query_vector = query_vector.numpy()
    else:
        # If it's already a NumPy array, no need to convert
        query_vector = np.array(query_vector)
    
    faiss.normalize_L2(query_vector)
    
    _, I = faiss_index.search(query_vector, top_k)
    
    context = ""
    for idx in I[0]:
        if idx in documents:
            _, sentence = documents[idx]
            context += sentence + " "
    
    if not context:
        # If no relevant context found, return the first few sentences of the document
        content_id = session.get('document_content_id', '')
        document_content = get_content_from_file(content_id) if content_id else ''
        sentences = sent_tokenize(document_content)
        context = ' '.join(sentences[:5])
    
    return context.strip()

def preprocess_text(text):
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def answer_question_local(question, context, is_highlighted):
    try:
        # Preprocess the question and context
        question = preprocess_text(question)
        context = preprocess_text(context)

        # Ensure we have enough context (up to 512 tokens)
        context = ' '.join(context.split()[:512])

        # Prepare the input for the model
        if is_highlighted:
            input_text = f"Highlighted text: {context}\n\nQuestion: {question}\nPlease answer the question based primarily on the highlighted text, but you may also use relevant information from the rest of the document if necessary."
        else:
            input_text = f"Context: {context}\n\nQuestion: {question}\nPlease answer the question based on the provided context."

        # Use the pipeline to get the answer
        result = qa_pipeline(question=question, context=input_text)
        
        # Extract the answer and confidence score
        answer = result['answer']
        confidence = result['score']
        
        # Format the response
        response = f"Answer: {answer} (Confidence: {confidence:.2f})"
        
        return response
    except Exception as e:
        print(f"Error in local model processing: {str(e)}")
        return f"An error occurred while processing your request: {str(e)}"

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def allowed_file(filename):
    """Check if the file extension is allowed."""
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'html', 'epub'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(file_path):
    """Extract text from a file based on its extension."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    print(f"Extracting text from: {file_path} with extension: {ext}")  # Debugging line

    if ext == '.pdf':
        return extract_pdf(file_path)
    elif ext == '.docx':
        return extract_docx(file_path)
    elif ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        print(f"Extracted text length: {len(content)}")  # Debugging line
        return content
    elif ext == '.html':
        with open(file_path, 'r', encoding='utf-8') as file:
            h = html2text.HTML2Text()
            h.ignore_links = True
            return h.handle(file.read())
    elif ext == '.epub':
        return extract_epub(file_path)
    else:
        return "Unsupported file format"

def extract_pdf(file_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        print(f"Extracted {len(text)} characters from PDF")
    except Exception as e:
        print(f"Error extracting PDF text: {str(e)}")
    return text

def extract_docx(file_path):
    """Extract text from a DOCX file."""
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_epub(file_path):
    """Extract text from an EPUB file."""
    book = open_book(file_path)
    lines = convert_epub_to_lines(book)
    return "\n".join(lines)



def answer_question_openai(question, context, is_highlighted):
    try:
        system_message = "You are a helpful assistant answering questions about a document."
        
        if is_highlighted:
            user_message = f"Highlighted text: {context}\n\nQuestion: {question}\n\nPlease answer the question based primarily on the highlighted text, but you may also use relevant information from the rest of the document if necessary."
        else:
            user_message = f"Context: {context}\n\nQuestion: {question}\n\nPlease answer the question based on the provided context."

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        return f"An error occurred while processing your request: {str(e)}"

def save_content_to_file(content):
    content_id = str(uuid.uuid4())
    file_path = os.path.join(tempfile.gettempdir(), f'doc_{content_id}.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return content_id

def get_content_from_file(content_id):
    file_path = os.path.join(tempfile.gettempdir(), f'doc_{content_id}.txt')
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

@app.route('/')
def index():
    """Render the index page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file
        file.save(file_path)
        print(f"Upload - File saved to: {file_path}")
        
        # Extract text from the file
        content = extract_text(file_path)
        print(f"Upload - Extracted content length: {len(content)}")
        
        if len(content) == 0:
            print(f"Warning: No content extracted from {filename}")
            flash('Failed to extract content from the file', 'error')
            return redirect(url_for('index'))
        
        # Generate a unique ID for the document
        doc_id = str(uuid.uuid4())
        
        # Preprocess and index the document
        preprocess_and_index_document(content, doc_id)
        
        # Save content to file and store ID in session
        content_id = save_content_to_file(content)
        session['document_content_id'] = content_id
        session['document_name'] = filename
        
        print(f"Upload - Document name set in session: {session.get('document_name')}")
        print(f"Upload - Document content ID set in session: {session.get('document_content_id')}")
        
        flash('Document uploaded and processed successfully', 'success')
        return redirect(url_for('chat'))
    
    flash('File type not allowed', 'error')
    return redirect(url_for('index'))


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    document_name = session.get('document_name', '')
    content_id = session.get('document_content_id', '')
    
    print(f"Chat - Document name from session: {document_name}")
    print(f"Chat - Document content ID from session: {content_id}")
    
    document_content = get_content_from_file(content_id) if content_id else ''
    print(f"Chat - Retrieved document content length: {len(document_content)}")
    
    # Check if the document is a PDF
    is_pdf = document_name.lower().endswith('.pdf')
    pdf_url = url_for('serve_pdf', filename=document_name) if is_pdf else None
    
    if request.method == 'POST':
        try:
            data = request.json
            question = data['question']
            model = data['model']
            is_highlighted = data.get('is_highlighted', False)
            
            print(f"Chat - Question: {question}")
            print(f"Chat - Model: {model}")
            print(f"Chat - Is highlighted: {is_highlighted}")
            
            if is_highlighted:
                context = f"Highlighted text: {data.get('context', '')} Context: {retrieve_relevant_context(question)}" 
            else:
                # Retrieve relevant context
                context = retrieve_relevant_context(question)
            
            print(f"Chat - Retrieved context length: {len(context)}")
            
            if model == 'openai':
                answer = answer_question_openai(question, context, is_highlighted)
            else:
                answer = answer_question_local(question, context, is_highlighted)
            
            print(f"Sending response: {answer}")
            
            return jsonify({'answer': answer})
        except Exception as e:
            print(f"Error in chat route: {str(e)}")
            return jsonify({'error': str(e)}), 500
    else:  # GET request
        return render_template('chat.html', 
                               document_name=document_name,
                               document_content=document_content,
                               is_pdf=is_pdf,
                               pdf_url=pdf_url)

@app.route('/uploads/<filename>')
def serve_pdf(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype='application/pdf')

@app.route('/clear_context', methods=['POST'])
def clear_context():
    # Clear any server-side context you might be storing
    # For example, you might want to reset the Faiss index or clear the documents dictionary
    global documents
    documents = {}
    global faiss_index
    faiss_index = faiss.IndexFlatL2(dimension)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    download_nltk_data()  # Ensure NLTK data is downloaded before running the app
    app.run( debug=True,host='0.0.0.0', port = 5001) #debug=True,
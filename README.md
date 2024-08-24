# Document Reader Helper(Academic Paper Reader)

This application is designed to help read and analyze academic papers. It provides an interface to upload documents (including PDFs), extract text, and ask questions about the content using both local and OpenAI-powered models.

## User Interface
![Application Interface](Screenshot%202024-08-24%20at%209.46.22%20AM.png)

## Features

- Upload and process various document types (PDF, DOCX, TXT, HTML, EPUB)
- Extract text from uploaded documents
- Use local models or OpenAI's API for question answering
- Display PDF files directly in the browser
- Context-aware responses based on document content

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Flask application:
   ```
   python app.py
   ```
2. Open a web browser and navigate to `http://localhost:5000`
3. Upload a document and start asking questions about its content

## Docker

Please note that the Dockerfile for this project is still a work in progress.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.




## Possible Improvements

1. Text Highlighting: Add the ability to highlight relevant parts of the document based on the AI's responses.
2. Multiple Document Support
3. Response Caching: Implement a caching mechanism to store and quickly retrieve answers for repeated questions.
4. Export Functionality: Allow users to export the chat history and relevant document excerpts.
5. Advanced Search: Implement a search feature within the document content.
6. Document Summarization: Add an option to generate a summary of the uploaded document.
7. Customizable AI Models: Allow users to choose or fine-tune specific AI models for their use case.


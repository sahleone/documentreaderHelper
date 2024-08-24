let storedSelectedText = '';
let highlightedRange = null;

function clearContext() {
    clearStoredSelection();
    fetch('/clear_context', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                console.log('Context cleared successfully');
                // Clear the chat messages
                $('#chat-messages').empty();
                // Clear the input field
                $('#user-input').val('');
            }
        })
        .catch(error => console.error('Error clearing context:', error));
}

/**
 * Capture the selected text and update the display.
 */
function captureSelectedText() {
    const selection = window.getSelection();
    const selectedText = selection.toString().trim();
    if (selectedText) {
        storedSelectedText = selectedText;
        updateSelectedTextDisplay();
        highlightSelection(selection);
    }
}

/**
 * Update the display of the selected text.
 */
function updateSelectedTextDisplay() {
    const displayElement = document.getElementById('selectedTextInfo');
    if (storedSelectedText) {
        displayElement.textContent = `Selected text: ${storedSelectedText.slice(0, 50)}${storedSelectedText.length > 50 ? '...' : ''}`;
        displayElement.style.display = 'block';
    } else {
        displayElement.textContent = '';
        displayElement.style.display = 'none';
    }
}

/**
 * Highlight the selected text in the document.
 */
function highlightSelection(selection) {
    if (highlightedRange) {
        highlightedRange.contents().unwrap();
    }
    if (selection.rangeCount > 0) {
        const range = selection.getRangeAt(0);
        const span = document.createElement('span');
        span.className = 'highlighted-text';
        range.surroundContents(span);
        highlightedRange = $(span);
    }
}

/**
 * Clear the stored selection and update the display.
 */
function clearStoredSelection() {
    storedSelectedText = '';
    updateSelectedTextDisplay();
    if (highlightedRange) {
        highlightedRange.contents().unwrap();
        highlightedRange = null;
    }
}

/**
 * Send a request to the server with the question and context.
 * @param {string} model - The model to use for the request.
 */
function sendRequest(model) {
    console.log("sendRequest function called with model:", model);
    const question = document.getElementById('question').value;
    const isHighlighted = storedSelectedText.length > 0;
    const content = isHighlighted ? storedSelectedText : document.getElementById('documentContent').textContent;
    
    if (!question.trim()) {
        alert("Please enter a question.");
        return;
    }

    console.log("Sending request with:", {
        question,
        contextLength: content.length,
        isHighlighted,
        model
    });

    fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            question: question,
            context: content,
            model: model,
            is_highlighted: isHighlighted
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log("Received response:", data);
        const chatbox = document.getElementById('chatbox');
        chatbox.innerHTML += `<p><strong>You:</strong> ${question}</p>`;
        chatbox.innerHTML += `<p><strong>AI (${model}):</strong> ${data.answer}</p>`;
        document.getElementById('question').value = '';
        chatbox.scrollTop = chatbox.scrollHeight;
    })
    .catch(error => {
        console.error('Error:', error);
        const chatbox = document.getElementById('chatbox');
        chatbox.innerHTML += `<p class="error"><strong>Error:</strong> ${error.message}</p>`;
        chatbox.scrollTop = chatbox.scrollHeight;
    });
}

function initializeDocument() {
    const documentContainer = document.querySelector('.document');
    if (documentContainer) {
        const pdfEmbed = documentContainer.querySelector('embed[type="application/pdf"]');
        if (pdfEmbed) {
            console.log("PDF detected, setting up PDF.js viewer");
            // Here you would set up PDF.js viewer if needed
        } else {
            documentContainer.addEventListener('mouseup', captureSelectedText);
            console.log("Event listener added to document container");
        }
    } else {
        console.log("Document container not found");
    }
}

function handleQuestionSubmit(event) {
    event.preventDefault();
    const model = event.target.dataset.model;
    sendRequest(model);
}

document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded and parsed");
    initializeDocument();
    
    const askLocalButton = document.getElementById('askLocalButton');
    const askOpenAIButton = document.getElementById('askOpenAIButton');
    const clearContextButton = document.getElementById('clearContextButton');
    
    if (askLocalButton) askLocalButton.addEventListener('click', handleQuestionSubmit);
    if (askOpenAIButton) askOpenAIButton.addEventListener('click', handleQuestionSubmit);
    if (clearContextButton) clearContextButton.addEventListener('click', clearContext);
});

document.addEventListener('DOMContentLoaded', function() {
    $('#user-input').keypress(function(e) {
        if (e.which == 13) {  // Enter key
            sendMessage();
        }
    });

    // Add event listener for the send button
    document.querySelector('button[onclick="sendMessage()"]').addEventListener('click', sendMessage);
});

function sendMessage() {
    var userInput = $('#user-input').val();
    if (userInput.trim() === '') return;

    $('#chat-messages').append('<p><strong>You:</strong> ' + userInput + '</p>');
    $('#user-input').val('');

    $.ajax({
        url: '/chat',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            question: userInput,
            model: 'openai'  // or 'local', depending on your preference
        }),
        success: function(response) {
            $('#chat-messages').append('<p><strong>AI:</strong> ' + response.answer + '</p>');
        },
        error: function(xhr, status, error) {
            $('#chat-messages').append('<p><strong>Error:</strong> ' + error + '</p>');
        }
    });
}

console.log("Script loaded");

document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded and parsed");
    initializeDocument();
    
    const askLocalButton = document.getElementById('askLocalButton');
    const askOpenAIButton = document.getElementById('askOpenAIButton');
    const clearContextButton = document.getElementById('clearContextButton');
    
    if (askLocalButton) {
        console.log("Ask Local button found");
        askLocalButton.addEventListener('click', handleQuestionSubmit);
    }
    if (askOpenAIButton) {
        console.log("Ask OpenAI button found");
        askOpenAIButton.addEventListener('click', handleQuestionSubmit);
    }
    if (clearContextButton) {
        console.log("Clear Context button found");
        clearContextButton.addEventListener('click', clearContext);
    }
});

function handleQuestionSubmit(event) {
    console.log("Question submitted");
    event.preventDefault();
    const model = event.target.dataset.model;
    sendRequest(model);
}

function sendRequest(model) {
    console.log(`Sending request to ${model} model`);
    const question = document.getElementById('question').value;
    const selectedText = document.getElementById('selectedTextInfo').textContent;
    
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            question: question,
            model: model,
            context: selectedText,
            is_highlighted: selectedText !== ''
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log("Response received:", data);
        document.getElementById('chatbox').innerHTML += `<p><strong>You:</strong> ${question}</p>`;
        document.getElementById('chatbox').innerHTML += `<p><strong>AI:</strong> ${data.answer}</p>`;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}
import gradio as gr
import ollama

# Function to get a response from Ollama (Mistral model)
def get_answer(question):
    try:
        # Send the question to the Mistral model via Ollama
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": question}])
        
        # Access the content field directly
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            return response.message.content  # Access the content field directly
        
        # If structure is unexpected
        return f"Response structure is different than expected. Full response: {response}"
    
    except Exception as e:
        # If an error occurs, print it out and return a user-friendly message
        return f"An error occurred: {str(e)}"

# Define custom CSS for styling
custom_css = """
    .output-text {
        font-size: 20px;
        height: 200px;
        resize: none;
        white-space: pre-wrap;
    }
    .input-text {
        font-size: 16px;
    }
    .gradio-container {
        display: flex;
        flex-direction: column-reverse;  /* This puts the output box above the input box */
    }
"""

# Create the Gradio interface
iface = gr.Interface(fn=get_answer, 
                     inputs=gr.Textbox(label="Ask a question:", lines=2),  # Text input box for the question
                     outputs=gr.Textbox(label="Answer:", lines=10, interactive=False, elem_id="output-box"),  # Bigger output box
                     title="Ollama Mistral Question Answering",  # Title of the interface
                     description="Ask any question, and I will get an answer from the Mistral model via Ollama.",  # Description
                     css=custom_css)  # Applying custom CSS to adjust layout and styling

# Launch the Gradio interface
iface.launch()

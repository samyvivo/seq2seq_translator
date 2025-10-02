import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the model and tokenizer
model_name = "jbochi/madlad400-3b-mt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

def translate_text(text, source_lang, target_lang):
    """
    Translate text between English and Persian using MADLAD-400-3B
    """
    # Define language codes for the model
    lang_codes = {
        "English": "en",
        "Persian": "fa"
    }
    
    source_code = lang_codes[source_lang]
    target_code = lang_codes[target_lang]
    
    # Create the translation prompt in the format the model expects
    prompt = f"<2{target_code}> {text}"
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Move inputs to the same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0
            )
        
        # Decode the output
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return translated_text
    
    except Exception as e:
        return f"Error during translation: {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="English-Persian Translator") as demo:
    gr.Markdown(
        """
        # 🌍 English-Persian Translator
        **Powered by MADLAD-400-3B Model**
        
        Translate text between English and Persian using the state-of-the-art MADLAD-400 model.
        """
    )
    
    with gr.Row():
        with gr.Column():
            source_lang = gr.Dropdown(
                choices=["English", "Persian"],
                value="English",
                label="Source Language"
            )
            input_text = gr.Textbox(
                lines=5,
                placeholder="Enter text to translate...",
                label="Input Text"
            )
            translate_btn = gr.Button("Translate", variant="primary")
        
        with gr.Column():
            target_lang = gr.Dropdown(
                choices=["Persian", "English"],
                value="Persian",
                label="Target Language"
            )
            output_text = gr.Textbox(
                lines=5,
                label="Translated Text",
                interactive=False
            )
    
    # Examples
    gr.Examples(
        examples=[
            ["Hello, how are you today?", "English", "Persian"],
            ["What is your name?", "English", "Persian"],
            ["سلام، حالتون چطوره؟", "Persian", "English"],
            ["امروز هوا خوب است", "Persian", "English"]
        ],
        inputs=[input_text, source_lang, target_lang],
        outputs=output_text,
        fn=translate_text,
        cache_examples=False
    )
    
    # Connect the button
    translate_btn.click(
        fn=translate_text,
        inputs=[input_text, source_lang, target_lang],
        outputs=output_text
    )
    
    # Auto-update target language based on source selection
    def update_target_lang(source_lang):
        return "Persian" if source_lang == "English" else "English"
    
    source_lang.change(
        fn=update_target_lang,
        inputs=source_lang,
        outputs=target_lang
    )

if __name__ == "__main__":
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        share=False,  # Set to True to get a public URL
        debug=True
    )
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from transformers import pipeline, AutoTokenizer
import io

def summarize_text(text, max_length=300, min_length=100):
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Error in summarization pipeline: {e}")
        return text

def answer_question(context, question):
    try:
        qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
        tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
        tokens = tokenizer(context, truncation=False, return_tensors="pt")["input_ids"]
        st.write(f"Context length: {len(context.split())} words")

        if len(tokens[0]) > 512:
            st.warning("Context is too long. Summarizing it...")
            context = summarize_text(context, max_length=300, min_length=100)
            st.write("Summary for QA:\n", context)
            
        result = qa_pipeline(question=question, context=context, truncation=True)
        return result['answer']
    except Exception as e:
        st.error(f"Error in QA pipeline: {e}")
        return "Could not generate an answer."

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text
    return text

def main():
    st.title("Document Summarization and Question Answering")
    st.sidebar.header("Upload Options")
    st.sidebar.write("Please upload a .pdf or .docx file to start processing.")
    
    uploaded_file = st.sidebar.file_uploader("Upload a document", type=["pdf", "docx"])
    
    text = ""
    if uploaded_file:
        # Extract text from file
        if uploaded_file.name.endswith(".pdf"):
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".docx"):
            text = extract_text_from_docx(uploaded_file)
        else:
            st.error("Unsupported file type!")
            return
        
    st.subheader("Extracted Text")
    st.text_area("Document Content", text, height=300)
    
    if st.button("Summarize Text"):
        with st.spinner("Summarizing text..."):
            summary_text = summarize_text(text)
            st.subheader("Summary")
            st.write(summary_text)
            
    question = st.text_input("Ask a question based on the document")
    if st.button("Get Answer"):
        with st.spinner("Finding the answer..."):
            answer = answer_question(text, question)
            st.subheader("Answer")
            st.write(answer)
            
    st.sidebar.info("This app uses Hugging Face's transformers library for summarization and question answering.")

if __name__ == "__main__":
    main()

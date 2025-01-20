from PyPDF2 import PdfReader
from docx import Document
from transformers import pipeline

        
def summarize_text(text, max_length=130,min_length=30):
    summarizer = pipeline("summarization",model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

def answer_question(question, context):
    try:
        qa_pipeline = pipeline("question-answering",model="deepset/roberta-base-squad2")
        if len(context) > 512:
            # context = summarize_text(context)
            raise ValueError("Context is too large for the model. Please summarize it first.")
        # qa_pipeline = pipeline("question-answering")
        result = qa_pipeline(question=question, context=context)
        return result['answer']
    except Exception as e:
        print(f"Error in QA pipeline: {e}")
        return "Could not generate an answer."

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text
    return text

# Test the functions
if __name__ == "__main__":
    file_path = r"T:\Me\Tejas_resume_Android_developer.pdf"  # Change to your file path
    if file_path.endswith(".pdf"):
        extract_text = extract_text_from_pdf(file_path)
        summarize_text = summarize_text(extract_text)
        print("\Original text:",extract_text)
        print("Summary:\n", summarize_text)
        
    elif file_path.endswith(".docx"):
        extract_text = extract_text_from_docx(file_path)
        summarize_text = summarize_text(extract_text)
        print("\Original text:",extract_text)
        print("Summary:\n", summarize_text)
    else:
        print("Unsupported file type!")
        
    context = extract_text
    question = "What projects has Tejas worked on?"
    answer = answer_question(context, question)
    print("\nQuestion:",question)
    print("Answer:", answer)
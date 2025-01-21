from PyPDF2 import PdfReader
from docx import Document
from transformers import pipeline, AutoTokenizer


def summarize_text(text, max_length = 300, min_length = 100):
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error in summarization pipeline: {e}")
        return text

def answer_question(context, question):
    try:
        qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
        tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
        tokens = tokenizer(context, truncation=False, return_tensors="pt")["input_ids"]
        print(f"Context length: {len(context.split())} words")

        # Skip summarization if context is within token limit
        if tokens.size(1) > 512:
            print("Context is too long. Summarizing it...")
            context = summarize_text(context, max_length=300, min_length=100)
            print("Summary for QA:\n", context)
            
        result = qa_pipeline(question=question, context=context, truncation=True)
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
        summary_text = summarize_text(extract_text, max_length=300, min_length=100)
        # print("Original text:\n", extract_text)
        # print("Summary:\n", summary_text)
        
    elif file_path.endswith(".docx"):
        extract_text = extract_text_from_docx(file_path)
        summary_text = summarize_text(extract_text, max_length=300, min_length=100)
        # print("Original text:\n", extract_text)
        # print("Summary:\n", summary_text)
    else:
        print("Unsupported file type!")
        exit(1)
        
            
    context = extract_text
    question = "what is tejas CGPA in Bachelor of Engineering?"
    answer = answer_question(context, question)
    print("\nQuestion:", question)
    print("Answer:", answer)

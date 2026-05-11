import sys

def extract_pdf(pdf_path):
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_docx(docx_path):
    import docx
    doc = docx.Document(docx_path)
    return "\n".join([p.text for p in doc.paragraphs])

if __name__ == "__main__":
    import json
    res = {}
    res['pdf'] = extract_pdf("SHL_AI_Intern_Assignment.pdf")
    res['docx'] = extract_docx("SHL_Assessment_Recommender_Architecture.docx")
    with open("docs_extracted.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("Done")

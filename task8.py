
import fitz  # PyMuPDF
import spacy, re, json

# Function to convert PDF to text
def pdf_to_text(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


nlp = spacy.load("en_core_web_sm")

# List of skills to detect
skills = ['python', 'machine learning', 'sql', 'tensorflow', 'django', 'flask', 'excel', 'react', 'javascript']

# Function to extract skills from text
def extract_skills(text):
    text = text.lower()
    found = {}
    for skill in skills:
        count = len(re.findall(r'\b' + re.escape(skill) + r'\b', text))
        if count > 0:
            found[skill] = count
    return found

# Function to calculate match score with job description
def match_score(resume_skills, job_text):
    job_text = job_text.lower()
    job_skills = [s for s in resume_skills if s in job_text]
    score = len(job_skills) / len(resume_skills) * 100
    return score



#  Or use a sample resume text (for testing)
resume_text = "Experienced Python and SQL developer with knowledge of Flask and Django."

#  Extract skills from resume
results = extract_skills(resume_text)

#  Print extracted skills
print("Extracted skills:", results)

#  Match with job description
job_description = "We need a Python developer with SQL and Flask experience."
print("Match score:", match_score(results, job_description), "%")

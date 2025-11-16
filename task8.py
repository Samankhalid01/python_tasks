
import fitz  #it comes form pyMuPDF it is used to read pdf files by this we can open pdf and extract text.
import spacy, re, json #re is python regular expression library we use it to search for patterns in texts this library is used to work with json data.

# Function to convert PDF to text
def pdf_to_text(file_path):
    text = "" #initialize empty string to store all pdfs text
    with fitz.open(file_path) as doc: #open the pdf file and doc is the object that represents the pdf file
        for page in doc: #iterate through each page of the pdf
            text += page.get_text()     #extract text from each page and append it to text string
    return text


nlp = spacy.load("en_core_web_sm") #spacy is a natural language processing library in python here we are loading the small english model

# List of skills to detect
skills = ['python', 'machine learning', 'sql', 'tensorflow', 'django', 'flask', 'excel', 'react', 'javascript']

# Function to extract skills from text
def extract_skills(text):
    text = text.lower() #convert text to lowercase .this ensures that Python and python are same skills
    found = {} #create empty dictionary to store found skills
    for skill in skills: #loop iterate thruogh each skill in skills list
        count = len(re.findall(r'\b' + re.escape(skill) + r'\b', text))#re.finall finds all occurrences of the skill in the text and returns them as a list len() counts how many times the skill appears in the text ,re.escape (skill) is used to escape any special characters in the skill string so that they are treated as literal characters in the regex pattern , hee r represents the usage of raw strings not the liternal meaning of backlash \n used in python
        if count > 0: #if skill atleast once in the text
            found[skill] = count #add skill and its count to found dictionary
    return found

# Function to calculate match score with job description
def match_score(resume_skills, job_text):
    job_text = job_text.lower()#make the job description text lowercase for case insensitive comparison
    job_skills = [s for s in resume_skills if s in job_text]#list comprehension to create a list of skills from resume_skills that are also present in job_text
    score = len(job_skills) / len(resume_skills) * 100 #calculate match score as percentage of resume skills found in job description
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

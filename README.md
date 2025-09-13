# AI Resume Screener

An AI-powered tool for screening resumes against job descriptions using Natural Language Processing (NLP).

## Features

- Upload multiple resumes (PDF/DOCX)
- Input or upload job descriptions
- Extract key information from resumes:
  - Name, email, phone
  - Skills
  - Education
  - Work experience
- AI-powered matching:
  - Skill matching
  - Overall text similarity
  - Match score calculation
- Visual analysis:
  - Skills match visualization
  - Skills word cloud
  - Detailed candidate analysis
- Export results to CSV

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

## Usage

1. **Upload Resumes**
   - Upload one or more resumes in PDF or DOCX format
   - The tool will automatically extract information from the resumes

2. **Input Job Description**
   - Paste the job description in the text area
   - Or upload a job description file (TXT, PDF, or DOCX)

3. **View Results**
   - See a summary table of all candidates with their match scores
   - Click on individual candidates to see detailed analysis
   - Download results as CSV for further processing

## Project Structure

```
resume_screening_tool/
├── app.py                   # Streamlit application
├── utils/
│   ├── parser.py            # Resume parsing logic
│   └── matcher.py           # Resume scoring logic
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Technologies Used

- Python
- Streamlit
- spaCy
- scikit-learn
- PyPDF2
- python-docx
- pandas
- matplotlib
- seaborn
- wordcloud

## Contributing

Feel free to submit issues and enhancement requests! 
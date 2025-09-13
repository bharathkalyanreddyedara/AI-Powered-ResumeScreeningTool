import streamlit as st
import pandas as pd
import base64
import tempfile
import os
from parser import ResumeParser
from matcher import ResumeMatcher
import matplotlib.pyplot as plt
import WordCloud
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import json
import sys

# --- Custom Fit Percentage Bar ---
def fit_percentage_bar(percentage, height=24):
    percentage = max(0, min(100, percentage))
    bar_html = f'''
    <div style="
        width: 100%;
        background: #353445;
        border-radius: 16px;
        height: {height}px;
        position: relative;
        margin-bottom: 8px;
        box-shadow: 0 2px 8px rgba(127,92,255,0.08);
    ">
        <div style="
            width: {percentage}%;
            background: linear-gradient(90deg, #7f5cff 0%, #00dbde 100%);
            height: 100%;
            border-radius: 16px 8px 8px 16px;
            transition: width 0.6s cubic-bezier(.4,0,.2,1);
            position: absolute;
            left: 0; top: 0;
        "></div>
        <span style="
            position: absolute;
            left: 50%; top: 50%;
            transform: translate(-50%, -50%);
            color: #fff;
            font-weight: 600;
            font-size: 1.1em;
            letter-spacing: 1px;
        ">{percentage:.1f}%</span>
    </div>
    '''
    st.markdown(bar_html, unsafe_allow_html=True)

# --- Advanced AI Tool CSS Styling ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
        background: linear-gradient(135deg, #232526 0%, #414345 100%);
        color: #e0e6ed;
    }
    .stApp {
        background: linear-gradient(120deg, #232526 0%, #414345 100%);
    }
    .glass {
        background: rgba(40, 44, 52, 0.7);
        border-radius: 18px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 2em 2em 1em 2em;
        margin-bottom: 2em;
    }
    .ai-glow {
        width:100%;height:100%;border-radius:50%;
        background:radial-gradient(circle, rgba(127,92,255,0.22) 0%, rgba(0,219,222,0.12) 70%, rgba(35,37,38,0.01) 100%);
        animation: aiPulse 2.5s infinite cubic-bezier(.4,0,.6,1);
    }
    @keyframes aiPulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.18); opacity: 0.85; }
        100% { transform: scale(1); opacity: 1; }
    }
    .ai-logo {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 80px;
        filter: drop-shadow(0 0 16px #7f5cff);
        animation: glow 2s infinite alternate;
    }
    @keyframes glow {
        from { filter: drop-shadow(0 0 8px #7f5cff); }
        to { filter: drop-shadow(0 0 32px #7f5cff); }
    }
    .section-title {
        font-size: 1.5em;
        font-weight: 700;
        letter-spacing: 1px;
        color: #7f5cff;
        margin-bottom: 0.5em;
        display: flex;
        align-items: center;
        gap: 0.5em;
    }
    .ai-badge {
        display: inline-block;
        padding: 0.3em 1em;
        border-radius: 16px;
        font-weight: 700;
        color: #fff;
        background: linear-gradient(90deg, #7f5cff 0%, #00dbde 100%);
        box-shadow: 0 2px 8px rgba(127, 92, 255, 0.2);
        margin-right: 0.5em;
        font-size: 1em;
        letter-spacing: 0.5px;
    }
    .ai-badge.low {background: linear-gradient(90deg, #e74c3c 0%, #ffb347 100%);}
    .ai-badge.medium {background: linear-gradient(90deg, #f1c40f 0%, #7f5cff 100%); color: #222;}
    .ai-badge.high {background: linear-gradient(90deg, #27ae60 0%, #00dbde 100%);}
    .ai-insight {
        background: rgba(127, 92, 255, 0.08);
        border-left: 4px solid #7f5cff;
        padding: 1em;
        border-radius: 10px;
        margin-top: 1em;
        color: #e0e6ed;
    }
    .stButton>button {
        color: white;
        background: linear-gradient(90deg, #7f5cff 0%, #00dbde 100%);
        border-radius: 8px;
        padding: 0.5em 1.5em;
        font-weight: 600;
        border: none;
    }
    .stFileUploader>div>div {
        border: 2px dashed #7f5cff !important;
        border-radius: 8px;
        background: rgba(127, 92, 255, 0.08);
    }
    .metric-label {font-size: 1.1em; color: #7f5cff;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar ---
with open("static/hirelens_logo.svg", "r") as f:
    svg_logo = f.read()
st.sidebar.markdown(
    f'''
    <div style="display:flex;justify-content:center;align-items:center;position:relative;height:120px;">
        <div style="position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:120px;height:120px;z-index:0;">
            <div class="ai-glow"></div>
        </div>
        <div style="background: #fff; border-radius: 50%; padding: 18px; box-shadow: 0 2px 12px rgba(127,92,255,0.10); width: 100px; height: 100px; display: flex; align-items: center; justify-content: center; position:relative; z-index:1;">
            {svg_logo}
        </div>
    </div>''',
    unsafe_allow_html=True)
st.sidebar.title("HireLens")
st.sidebar.markdown("<span style='color:#7f5cff;font-weight:600;'>Paste or type the job description below, or upload a file in the main area.</span>", unsafe_allow_html=True)

# --- Trainable Mock AI Agent ---
MAPPING_FILE = "job_title_skills_mapping.json"

def load_skills_mapping():
    try:
        with open(MAPPING_FILE, "r") as f:
            loaded = json.load(f)
    except Exception:
        # Default mapping
        loaded = {
            "data scientist": ["tableau", "power BI", "python", "machine learning", "data analysis", "statistics", "pandas", "numpy", "scikit-learn", "sql", "deep learning", "visualization"],
            "data analyst": ["excel", "sql", "python", "power BI", "tableau", "data visualization", "data cleaning", "statistics", "business analysis", "pandas", "numpy"],
            "software developer": ["java", "python", "c++", "data structures", "algorithms", "git", "debugging", "oop", "sql", "rest APIs", "unit testing", "html", "css", "javascript", "react", "node.js", "express.js", "mongodb", "docker", "CI/CD", "aws"],
            "full stack developer": ["html", "css", "javascript", "react", "node.js", "express.js", "mongodb", "sql", "python", "django", "flask", "git", "docker", "rest API", "CI/CD", "aws"],
            "frontend developer": ["html", "css", "javascript", "react", "angular", "vue.js", "bootstrap", "tailwind CSS", "webpack", "npm", "responsive design"],
            "backend developer": ["node.js", "express.js", "django", "flask", "spring boot", "java", "python", "sql", "mongodb", "rest API", "jwt", "authentication", "docker"],
            "machine learning engineer": ["python", "machine learning", "scikit-learn", "pandas", "numpy", "tensorflow", "pytorch", "deep learning", "flask", "fastAPI", "model deployment", "mlops"],
            "AI engineer": ["python", "deep learning", "tensorflow", "pytorch", "nlp", "computer vision", "transformers", "keras", "scikit-learn", "model optimization"],
            "devops engineer": ["linux", "bash", "python", "git", "jenkins", "docker", "kubernetes", "ansible", "terraform", "aws", "azure", "gcp", "prometheus", "grafana", "CI/CD"],
            "cloud engineer": ["aws", "azure", "gcp", "ec2", "s3", "cloud networking", "iam", "cloud security", "docker", "kubernetes", "terraform", "cloudformation", "devops tools"],
            "cybersecurity analyst": ["network security", "firewalls", "ids/ips", "penetration testing", "siem", "incident response", "splunk", "nessus", "wireshark", "compTIA security+", "ethical hacking"],
            "network engineer": ["tcp/ip", "routing", "switching", "dns", "dhcp", "ospf", "bgp", "network troubleshooting", "cisco", "ccna", "vpn", "firewalls", "load balancers"],
            "system administrator": ["linux", "windows server", "user management", "bash scripting", "powershell", "network configuration", "vmware", "backup", "monitoring", "patch management"],
            "database administrator": ["mysql", "postgresql", "oracle", "sql server", "database tuning", "backup and recovery", "replication", "indexing", "normalization", "nosql", "mongodb"],
            "QA engineer": ["manual testing", "automation testing", "selenium", "cypress", "jmeter", "postman", "bug tracking", "test cases", "CI/CD", "regression testing", "unit testing"],
            "UI/UX designer": ["figma", "adobe xd", "sketch", "wireframing", "prototyping", "usability testing", "interaction design", "html", "css", "responsive design", "design systems"],
            "product manager": ["agile", "scrum", "roadmapping", "user stories", "market research", "sql", "jira", "communication", "product lifecycle", "analytics"],
            "IT support": ["hardware troubleshooting", "windows", "linux", "networking", "active directory", "ticketing systems", "customer support", "remote tools", "diagnostics"],
            "blockchain developer": ["solidity", "ethereum", "smart contracts", "web3.js", "blockchain architecture", "dapps", "cryptography", "truffle", "ganache", "hyperledger"],
            "game developer": ["c++", "c#", "unity", "unreal engine", "game physics", "3d math", "graphics programming", "animation", "shader programming"],
            "mobile app developer": ["android", "ios", "flutter", "react native", "kotlin", "swift", "firebase", "api integration", "ui/ux design", "play store deployment"],
            "site reliability engineer": ["linux", "python", "monitoring", "incident response", "prometheus", "grafana", "kubernetes", "ci/cd", "sre principles"],
            "data engineer": ["hadoop", "spark", "sql", "python", "etl", "airflow", "data pipelines", "bigquery", "redshift", "snowflake"],
            "business intelligence developer": ["power BI", "tableau", "ssrs", "ssas", "sql", "data warehousing", "dashboards", "data modeling"],
            "rpa developer": ["uipath", "automation anywhere", "blue prism", "workflow automation", "vb.net", "bot deployment"],
            "technical writer": ["technical documentation", "markdown", "api docs", "uml", "tools like confluence", "html", "editorial skills"],
            "scrum master": ["agile", "scrum", "jira", "kanban", "team facilitation", "retrospectives", "burn down charts"],
            "solutions architect": ["cloud architecture", "aws", "azure", "system design", "microservices", "security architecture", "cost optimization", "design patterns"],
            "iot developer": ["arduino", "raspberry pi", "iot protocols", "c", "c++", "python", "sensors", "embedded systems", "mqtt", "iot cloud platforms"],
            "devrel / developer advocate": ["public speaking", "community engagement", "technical writing", "content creation", "open source", "api demos", "github"],
            "localization engineer": ["i18n", "l10n", "translation memory", "unicode", "localization tools", "xml", "json", "internationalization"],
            "quantum computing researcher": ["quantum algorithms", "qiskit", "quantum circuits", "linear algebra", "python", "quantum mechanics"],
            "accessibility engineer": ["a11y standards", "aria roles", "wcag", "screen readers", "html", "css", "javascript accessibility libraries"],
            "AI research scientist": ["deep learning", "research papers", "model development", "python", "tensorflow", "pytorch", "nlp", "cv", "optimization", "scientific writing"],
            "embedded systems engineer": ["c", "c++", "embedded C", "microcontrollers", "real-time systems", "rtos", "spi", "i2c", "uart", "debugging tools"],
            "digital forensics analyst": ["forensics tools", "disk imaging", "memory analysis", "log analysis", "siem", "wireshark", "report writing", "chain of custody"],
            "game designer": ["game mechanics", "storyboarding", "level design", "unity", "unreal engine", "prototyping", "user experience", "balancing"],
            "hardware engineer": ["circuit design", "pcb layout", "microprocessors", "fpga", "verilog", "vhdl", "digital electronics", "simulation tools"],
            "virtualization engineer": ["vmware", "hyper-v", "virtualbox", "kvm", "proxmox", "virtual networking", "storage management", "scripting"],
            "web developer": ["html", "css", "javascript", "react", "vue.js", "seo", "accessibility", "wordpress", "api integration"],
            "test automation engineer": ["selenium", "cypress", "robot framework", "testng", "pytest", "jenkins", "ci/cd", "page object model"],
            "platform engineer": ["linux", "cloud infrastructure", "container orchestration", "sre practices", "infrastructure as code", "observability tools"],
            "chief technology officer": ["technology strategy", "leadership", "architecture review", "budgeting", "team scaling", "stakeholder communication"],
            "technical program manager": ["agile", "roadmap planning", "cross-functional leadership", "risk management", "jira", "technical architecture"],
            "cloud security engineer": ["iam", "encryption", "security groups", "firewalls", "cloudtrail", "security compliance", "cloudwatch", "incident response"],
            "ETL developer": ["etl tools", "data warehousing", "sql", "data transformation", "airflow", "informatica", "talend", "data staging"],
            "observability engineer": ["logs", "metrics", "traces", "prometheus", "grafana", "datadog", "splunk", "openTelemetry", "dashboarding"],
            "AI ethics analyst": ["bias detection", "data fairness", "model explainability", "regulatory compliance", "ai ethics frameworks", "python"],
            "bioinformatics engineer": ["biopython", "r", "genomics", "data analysis", "pipeline development", "bioinformatics tools", "statistics"],
            "augmented reality developer": ["unity", "arcore", "arkit", "3d modeling", "c#", "scene management", "animation", "camera access"],
            "robotics engineer": ["ros", "python", "c++", "robot kinematics", "machine vision", "embedded systems", "actuators", "path planning"],
            "fintech developer": ["python", "java", "api integration", "blockchain", "data security", "financial protocols", "sql", "regulatory compliance"]
        }
    # Normalize keys and skill values
    normalized = {}
    for k, v in loaded.items():
        try:
            key_norm = k.lower().strip()
        except Exception:
            continue
        if isinstance(v, list):
            normalized[key_norm] = [str(s).strip() for s in v]
        else:
            normalized[key_norm] = v
    return normalized

def save_skills_mapping(mapping):
    with open(MAPPING_FILE, "w") as f:
        json.dump(mapping, f, indent=2)

skills_mapping = load_skills_mapping()

# --- Sidebar ---
st.sidebar.subheader("Job Title 🏷️")
job_title = st.sidebar.text_input("Enter the job title: ")

if 'required_skills' not in st.session_state:
    st.session_state['required_skills'] = []

job_title_key = job_title.lower().strip() if job_title else ""

# If job title is in mapping, use it
if job_title_key in skills_mapping:
    required_skills = skills_mapping[job_title_key]
    st.session_state['required_skills'] = required_skills
else:
    required_skills = st.session_state['required_skills']

# If job title is not in mapping, allow user to add skills
if job_title and job_title_key not in skills_mapping:
    st.sidebar.markdown("<b>No skills found for this job title. Please enter a comma-separated list of required skills:</b>", unsafe_allow_html=True)
    new_skills_input = st.sidebar.text_area("Required Skills (comma-separated)", value=", ".join(required_skills) if required_skills else "", height=80)
    if st.sidebar.button("Save Skills for This Title"):
        new_skills = [s.strip() for s in new_skills_input.split(",") if s.strip()]
        if new_skills:
            skills_mapping[job_title_key] = new_skills
            save_skills_mapping(skills_mapping)
            st.session_state['required_skills'] = new_skills
            st.sidebar.success(f"Skills saved for '{job_title}'.")
            required_skills = new_skills

if required_skills:
    st.sidebar.markdown("<b>Required Skills:</b>", unsafe_allow_html=True)
    skills_str = ", ".join(required_skills)
    st.sidebar.markdown(f"<div style='color:#7f5cff; font-size:1.1em; margin-bottom:1em;'>{skills_str}</div>", unsafe_allow_html=True)

st.sidebar.subheader("Job Description 📝")
job_desc_text_input = st.sidebar.text_area(
    "Paste or type the job description here",
    height=200,
    help="You can paste or type the job description."
)

# --- Main Area ---
with open("static/hirelens_logo.svg", "r") as f:
    svg_logo_main = f.read()
st.markdown(
    f'''
    <div style="display:flex;justify-content:center;align-items:center;position:relative;height:150px;">
        <div style="position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:160px;height:160px;z-index:0;">
            <div class="ai-glow"></div>
        </div>
        <div style="background: #fff; border-radius: 50%; padding: 24px; box-shadow: 0 2px 16px rgba(127,92,255,0.12); width: 120px; height: 120px; display: flex; align-items: center; justify-content: center; position:relative; z-index:1;">
            {svg_logo_main}
        </div>
    </div>''',
    unsafe_allow_html=True)
st.title("🤖 HireLens: Advanced AI Resume Screening Dashboard")
st.markdown(
    """
    <div style='font-size:1.1em; color:#e0e6ed;'>
    <b>AI-powered</b> resume screening and candidate shortlisting.<br>
    Paste or type your job description in the sidebar, or upload a file below. Then upload resumes to begin.
    </div>
    """,
    unsafe_allow_html=True
)

# --- Job Description File Upload (Main Area) ---
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.subheader("Upload Job Description File(s)")
job_desc_files = st.file_uploader(
    "Upload one or more Job Descriptions (PDF, DOCX, TXT, Images)",
    type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
    accept_multiple_files=True,
    help="You can upload one or more job description files."
)
st.markdown("</div>", unsafe_allow_html=True)

# --- Resume Upload Section ---
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📤 Upload Resumes</div>", unsafe_allow_html=True)
resume_files = st.file_uploader(
    "Drag and drop or select one or more resumes (PDF/DOCX/Images)",
    type=["pdf", "docx", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
    accept_multiple_files=True,
    help="You can upload one or more resumes."
)
st.markdown("</div>", unsafe_allow_html=True)

# --- Screening Trigger Button with Session State ---
if 'screening_triggered' not in st.session_state:
    st.session_state['screening_triggered'] = False

def trigger_screening():
    st.session_state['screening_triggered'] = True

st.button("🔍 Perform Screening", on_click=trigger_screening)

# --- Load Models ---
@st.cache_resource
def load_models():
    return ResumeParser(), ResumeMatcher()
parser, matcher = load_models()

# --- Process Inputs ---
# Parse job descriptions
job_desc_texts = []
job_desc_names = []
if job_desc_files:
    for file in job_desc_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name
        if file.name.endswith('.pdf'):
            text = parser.extract_text_from_pdf(tmp_file_path)
        elif file.name.endswith('.docx'):
            text, _ = parser.extract_text_from_docx(tmp_file_path)
        elif file.name.endswith('.txt'):
            text = file.getvalue().decode()
        elif file.name.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp")):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
            if parser.is_tesseract_available():
                text = parser.extract_text_from_image(tmp_file_path)
            else:
                st.warning("Image OCR is unavailable because Tesseract OCR is not installed. Install it to process images.")
                text = None
            os.unlink(tmp_file_path)
        else:
            text = None
        if text:
            job_desc_texts.append(text)
            job_desc_names.append(file.name)
        os.unlink(tmp_file_path)
# Fallback: use sidebar text area if no files
elif job_desc_text_input and job_desc_text_input.strip():
    job_desc_texts = [job_desc_text_input.strip()]
    job_desc_names = ["Manual Input"]

# --- Screening Logic ---
if st.session_state['screening_triggered'] and job_desc_texts and resume_files:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 Screening Results</div>", unsafe_allow_html=True)
    results = []
    progress = st.progress(0, text="Analyzing resumes with AI...")
    total = len(resume_files) * len(job_desc_texts)
    count = 0
    for job_idx, (job_desc_text, job_desc_name) in enumerate(zip(job_desc_texts, job_desc_names)):
        for resume in resume_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(resume.name)[1]) as tmp_file:
                tmp_file.write(resume.getvalue())
                resume_path = tmp_file.name
            if resume.name.endswith('.pdf') or resume.name.endswith('.docx'):
                resume_data = parser.parse_resume(resume_path)
                score_data = matcher.score_resume(resume_data, job_desc_text)
                print('==== DEBUG: Resume Data ====')
                print(resume_data)
                print('==== DEBUG: Score Data ====')
                print(score_data)
                print('===========================')
            elif resume.name.endswith('.txt'):
                text = resume.getvalue().decode()
                resume_data = {'raw_text': text, 'name': '', 'email': '', 'phone': '', 'skills': [], 'education': [], 'experience': [], 'links': []}
                score_data = matcher.score_resume(resume_data, job_desc_text)
            elif resume.name.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp")):
                if parser.is_tesseract_available():
                    text = parser.extract_text_from_image(resume_path)
                    resume_data = {'raw_text': text, 'name': '', 'email': '', 'phone': '', 'skills': parser.extract_skills(text), 'education': parser.extract_education(text), 'experience': parser.extract_experience(text), 'links': parser.extract_links(text)}
                    score_data = matcher.score_resume(resume_data, job_desc_text)
                else:
                    st.warning("Image OCR is unavailable because Tesseract OCR is not installed. Install it to process images.")
                    resume_data = None
                    score_data = None
            else:
                resume_data = None
                score_data = None
            # Only append results if resume_data and score_data are valid
            if resume_data is not None and score_data is not None:
                # Skill match with required_skills if present
                if required_skills:
                    resume_skills = set([s.lower() for s in resume_data['skills']])
                    req_skills = set([s.lower() for s in required_skills])
                    matched_skills = list(resume_skills & req_skills)
                    missing_skills = list(req_skills - resume_skills)
                    skill_match_pct = (len(matched_skills) / len(req_skills) * 100) if req_skills else 0
                else:
                    matched_skills = []
                    missing_skills = []
                    skill_match_pct = None
                results.append({
                    'Job Description': job_desc_name,
                    'Resume': resume.name,
                    'Name': resume_data['name'],
                    'Email': resume_data['email'],
                    'Match Score': score_data['final_score'],
                    'Skill Match %': skill_match_pct,
                    'Matched Skills': ', '.join(matched_skills) if matched_skills else '',
                    'Missing Skills': ', '.join(missing_skills) if missing_skills else '',
                    'Resume Data': resume_data,
                    'Score Data': score_data,
                    'Job Desc Text': job_desc_text
                })
            os.unlink(resume_path)
            count += 1
            progress.progress(count/total, text=f"AI analyzed {count}/{total} combinations")
    progress.empty()

    # --- Results DataFrame ---
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['Resume Data', 'Score Data', 'Job Desc Text']} for r in results])
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown(f'<a href="data:file/csv;base64,{b64}" download="screening_results.csv">⬇️ Download Results as CSV</a>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Candidate Analysis Section ---
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🧠 AI Candidate Analysis</div>", unsafe_allow_html=True)
    for r in results:
        with st.expander(f"{r['Name'] or r['Resume']} vs {r['Job Description']} - ", expanded=False):
            col1, col2 = st.columns([1,2])
            with col1:
                score = r['Match Score']
                if score >= 70:
                    badge_class = 'high'
                    badge_icon = '🤖'
                elif score >= 40:
                    badge_class = 'medium'
                    badge_icon = '🧠'
                else:
                    badge_class = 'low'
                    badge_icon = '⚠️'
                st.markdown(f'<span class="ai-badge {badge_class}">{badge_icon} AI Confidence: {score:.2f}%</span>', unsafe_allow_html=True)
                fit_percentage_bar(score)
                if r['Skill Match %'] is not None:
                    st.metric("Skill Match %", f"{r['Skill Match %']:.2f}%")
                st.metric("Overall Similarity", f"{r['Score Data']['overall_similarity']:.2f}%")
                st.markdown(f"**Email:** {r['Email'] or 'N/A'}")
                if r['Skill Match %'] is not None:
                    st.markdown(f"**Matched Skills:** <span style='color:#27ae60'>{r['Matched Skills'] or 'None'}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Missing Skills:** <span style='color:#e74c3c'>{r['Missing Skills'] or 'None'}</span>", unsafe_allow_html=True)
                st.markdown(f"**Education:** <br>{'<br>'.join(r['Resume Data']['education']) or 'N/A'}", unsafe_allow_html=True)
                st.markdown(f"**Experience:** <br>{'<br>'.join(r['Resume Data']['experience']) or 'N/A'}", unsafe_allow_html=True)
                st.markdown(f"<b>Job Description:</b> <br><span style='color:#7f5cff'>{r['Job Description']}</span>", unsafe_allow_html=True)
                # --- AI Insights ---
                st.markdown("<div class='ai-insight'>", unsafe_allow_html=True)
                if score >= 70:
                    st.markdown("<b>AI Insight:</b> This candidate is an <span style='color:#27ae60'>excellent fit</span> for the job description. Highly recommended for shortlisting.", unsafe_allow_html=True)
                elif score >= 40:
                    st.markdown("<b>AI Insight:</b> This candidate is a <span style='color:#f1c40f'>potential fit</span>. Review their skills and experience for further consideration.", unsafe_allow_html=True)
                else:
                    st.markdown("<b>AI Insight:</b> This candidate has a <span style='color:#e74c3c'>low match</span> with the job description. Consider only if other criteria are met.", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                st.markdown("**Skills Word Cloud**")
                wordcloud = WordCloud(width=600, height=300, background_color='white', colormap='cool').generate(r['Resume Data']['raw_text'])
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)
    # Reset the trigger so user can screen again
    st.session_state['screening_triggered'] = False
else:
    st.markdown("<div class='glass' style='text-align:center;'><span style='color:#7f5cff'>Please upload at least one job description (file or text) and at least one resume, then click 'Perform Screening' to begin AI screening.</span></div>", unsafe_allow_html=True)

print("Python executable:", sys.executable)

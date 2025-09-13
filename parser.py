import docx2txt
from docx2python import docx2python
import re
from typing import Dict, List, Union, Optional, Tuple
import spacy
from pathlib import Path
import pdfplumber
import zipfile
from PIL import Image
import pytesseract
import os
import platform
from shutil import which

# Try to auto-configure Tesseract path on Windows if not on PATH
if platform.system().lower().startswith('win'):
    if which('tesseract') is None:
        common_paths = [
            r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
            r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"
        ]
        for p in common_paths:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break

class ResumeParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    @staticmethod
    def is_tesseract_available() -> bool:
        try:
            # If pytesseract has an explicit cmd configured, ensure it exists
            tcmd = getattr(pytesseract.pytesseract, 'tesseract_cmd', None)
            if tcmd:
                return os.path.exists(tcmd)
            return which('tesseract') is not None
        except Exception:
            return False

    def extract_hyperlinks_from_docx(self, file_path: str) -> List[str]:
        """Extract all hyperlinks (including mailto:) from a DOCX file using docx2python, with raw XML fallback."""
        doc_result = docx2python(file_path)
        links = set()
        hyperlink_map = getattr(doc_result, 'hyperlink_map', None)
        if hyperlink_map is None and hasattr(doc_result, 'properties'):
            hyperlink_map = doc_result.properties.get('hyperlink_map', {})
        if isinstance(hyperlink_map, dict):
            for url in hyperlink_map.values():
                links.add(url)
        # Fallback: extract all URLs from raw XML if no links found
        if not links:
            with zipfile.ZipFile(file_path) as docx_zip:
                for name in docx_zip.namelist():
                    if name.endswith('.xml'):
                        xml_content = docx_zip.read(name).decode(errors='ignore')
                        url_pattern = r'(https?://\S+|www\.\S+|mailto:[^\s\)\]\[\"\'>]+)'
                        for match in re.findall(url_pattern, xml_content):
                            links.add(match)
        return list(links)

    def extract_text_from_pdf(self, file_path: str) -> str:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    def extract_text_from_image(self, file_path: str) -> str:
        """Extract text from an image using Tesseract OCR."""
        try:
            with Image.open(file_path) as img:
                img_converted = img.convert('L')
                text = pytesseract.image_to_string(img_converted)
                return text or ""
        except pytesseract.TesseractNotFoundError as e:
            print("WARN: Tesseract OCR not found. Install from https://github.com/UB-Mannheim/tesseract/wiki and add to PATH.")
            return ""
        except Exception:
            return ""

    def extract_text_from_docx(self, file_path: str) -> Tuple[str, List[str]]:
        text = docx2txt.process(file_path)
        links = self.extract_hyperlinks_from_docx(file_path)
        # Append all links to the text to ensure they are available for extraction
        if links:
            text += '\n' + '\n'.join(links)
        print('==== DEBUG: DOCX Extracted Links ====')
        print(links)
        print('====================================')
        return text, links

    def extract_email(self, text: str, links: Optional[List[str]] = None) -> str:
        if links is None:
            links = []
        email_pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
        # 1. Find all emails in text
        emails = re.findall(email_pattern, text)
        # 2. Find all emails in links (e.g., mailto:)
        for link in links:
            if link.startswith('mailto:'):
                email = link[7:]
                if re.match(email_pattern, email):
                    emails.append(email)
        # 3. Obfuscated forms (e.g., name [at] domain.com)
        obf_pattern = r'([A-Za-z0-9._%+-]+)\s*\[at\]\s*([A-Za-z0-9.-]+)\s*\[dot\]\s*([A-Za-z]{2,})'
        obf_match = re.search(obf_pattern, text, re.IGNORECASE)
        if obf_match:
            emails.append(f"{obf_match.group(1)}@{obf_match.group(2)}.{obf_match.group(3)}")
        print('==== DEBUG: All Found Emails ====')
        print(emails)
        print('==== DEBUG: All Links ====')
        print(links)
        print('================================')
        return emails[0] if emails else ""

    def extract_phone(self, text: str) -> str:
        phone_pattern = r'\+?[\d\s-]{10,}'
        match = re.search(phone_pattern, text)
        return match.group(0) if match else ""

    def extract_name(self, text: str) -> str:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        top_text = '\n'.join(lines[:40])
        doc = self.nlp(top_text)
        contact_keywords = ['gmail', 'linkedin', 'email', 'phone', 'contact', 'address', 'resume', 'curriculum', 'cv']
        candidates = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        filtered = [
            c for c in candidates
            if 1 < len(c.split()) < 5
            and not any(keyword in c.lower() for keyword in contact_keywords)
            and not any(char.isdigit() for char in c)
            and not any(sym in c for sym in '@|♦•')
        ]
        if filtered:
            return filtered[0]
        for line in lines[:40]:
            if (
                1 < len(line.split()) < 5 and
                not any(keyword in line.lower() for keyword in contact_keywords) and
                not any(char.isdigit() for char in line) and
                not any(sym in line for sym in '@|♦•') and
                all(w.isalpha() for w in line.split())
            ):
                if not (line.isupper() and not all(len(w) <= 2 for w in line.split())):
                    return line
        for line in lines[:40]:
            if all(w.isalpha() for w in line.split()) and not any(keyword in line.lower() for keyword in contact_keywords):
                return line
        return lines[0] if lines else ""

    def extract_skills(self, text: str) -> List[str]:
        skill_keywords = [
            'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue', 'node.js', 'nodejs', 'express', 'django', 'flask', 'spring', 'ruby', 'php', 'swift', 'kotlin', 'go', 'rust', 'c++', 'c#',
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'elasticsearch', 'dynamodb',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'ci/cd', 'devops', 'agile', 'scrum',
            'machine learning', 'deep learning', 'ai', 'nlp', 'computer vision', 'data science', 'data analysis', 'big data',
            'html', 'css', 'sass', 'less', 'bootstrap', 'tailwind', 'jquery', 'redux', 'graphql', 'rest', 'api'
        ]
        text_lower = text.lower()
        found_skills = set()
        for skill in skill_keywords:
            if ' ' in skill or '/' in skill or '.' in skill or '+' in skill or '#' in skill:
                if skill in text_lower:
                    found_skills.add(skill)
            else:
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower):
                    found_skills.add(skill)
        return sorted(found_skills)

    def extract_education(self, text: str) -> List[str]:
        education_keywords = ['bachelor', 'master', 'phd', 'bsc', 'msc', 'mba', 'b.tech', 'm.tech']
        education = []
        lines = text.lower().split('\n')
        for line in lines:
            if any(keyword in line for keyword in education_keywords):
                education.append(line.strip())
        return education

    def extract_experience(self, text: str) -> List[str]:
        experience_keywords = ['experience', 'work', 'employment', 'job']
        experience = []
        lines = text.lower().split('\n')
        for i, line in enumerate(lines):
            if any(keyword in line for keyword in experience_keywords):
                exp_text = ' '.join(lines[i:i+5])
                experience.append(exp_text.strip())
        return experience

    def extract_links(self, text: str) -> List[str]:
        url_pattern = r'(https?://\S+|www\.\S+|mailto:[^\s\)\]\[\"\'>]+)'
        return re.findall(url_pattern, text)

    def extract_links_from_pdf(self, file_path: str) -> List[str]:
        print(f"==== DEBUG: Called extract_links_from_pdf on {file_path} ====")
        import pdfplumber
        links = set()
        with pdfplumber.open(file_path) as pdf:
            print(f"==== DEBUG: PDF has {len(pdf.pages)} pages ====")
            for i, page in enumerate(pdf.pages):
                print(f"==== DEBUG: PDF Page {i+1} Attributes ====")
                print("hyperlinks:", getattr(page, 'hyperlinks', None))
                print("annots:", getattr(page, 'annots', None))
                print("raw dict:", page.to_dict())
                print("==========================================")
                # 1. pdfplumber's hyperlinks property (most reliable)
                if hasattr(page, 'hyperlinks') and page.hyperlinks:
                    for h in page.hyperlinks:
                        if 'uri' in h:
                            links.add(h['uri'])
                # 2. Annotations (older pdfplumber versions)
                if hasattr(page, 'annots') and page.annots:
                    for annot in page.annots:
                        uri = annot.get('uri')
                        if uri:
                            links.add(uri)
                # 3. Regex search in text
                text = page.extract_text() or ''
                url_pattern = r'(https?://\S+|www\.\S+|mailto:[^\s\)\]\[\"\'>]+)'
                for match in re.findall(url_pattern, text):
                    links.add(match)
        print("==== DEBUG: FINAL EXTRACTED LINKS ====")
        print(list(links))
        print("======================================")
        return list(links)

    def dump_docx_xml(self, file_path: str):
        import zipfile
        print("==== DOCX XML DUMP ====")
        with zipfile.ZipFile(file_path) as docx_zip:
            for name in docx_zip.namelist():
                if name.endswith('.xml'):
                    print(f"--- {name} ---")
                    xml_content = docx_zip.read(name).decode(errors='ignore')
                    print(xml_content[:2000])  # Print first 2000 chars for brevity
        print("=======================")

    def parse_resume(self, file_path: Union[str, Path]) -> Dict:
        print(f"==== DEBUG: parse_resume called on {file_path} ====")
        file_path = str(file_path)
        if file_path.lower().endswith('.pdf'):
            text = self.extract_text_from_pdf(file_path)
            links = self.extract_links_from_pdf(file_path)
            print(f"==== DEBUG: Links after extract_links_from_pdf: {links} ====")
        elif file_path.lower().endswith('.docx'):
            self.dump_docx_xml(file_path)
            text, docx_links = self.extract_text_from_docx(file_path)
            links = list(set(self.extract_links(text)) | set(docx_links))
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp')):
            text = self.extract_text_from_image(file_path)
            links = self.extract_links(text)
        else:
            raise ValueError("Unsupported file format. Please upload PDF, DOCX, or image files.")
        print("==== Resume Extracted Text Preview ====")
        for i, line in enumerate(text.split('\n')[:10]):
            print(f"{i+1}: {line}")
        print("======================================")
        print("==== DEBUG: Extracted Text ====")
        print(text)
        print("==== DEBUG: Extracted Links ====")
        print(links)
        print("================================")
        return {
            'name': self.extract_name(text),
            'email': self.extract_email(text, links),
            'phone': self.extract_phone(text),
            'skills': self.extract_skills(text),
            'education': self.extract_education(text),
            'experience': self.extract_experience(text),
            'links': links,
            'raw_text': text
        } 
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import numpy as np
import re

class ResumeMatcher:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text by removing special characters and converting to lowercase."""
        doc = self.nlp(text.lower())
        # Remove stop words and lemmatize
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return ' '.join(tokens)
    
    def extract_skills_from_job_description(self, job_description: str) -> List[str]:
        """Extract required skills from job description."""
        # Common technical skills to look for
        skill_patterns = [
            r'\b(python|java|javascript|typescript|react|angular|vue|node\.js|express|django|flask|spring|ruby|php|swift|kotlin|go|rust|c\+\+|c#)\b',
            r'\b(sql|mysql|postgresql|mongodb|redis|cassandra|elasticsearch|dynamodb)\b',
            r'\b(aws|azure|gcp|docker|kubernetes|jenkins|git|ci/cd|devops|agile|scrum)\b',
            r'\b(machine learning|deep learning|ai|nlp|computer vision|data science|data analysis|big data)\b',
            r'\b(html|css|sass|less|bootstrap|tailwind|jquery|redux|graphql|rest|api)\b'
        ]
        
        skills = set()
        for pattern in skill_patterns:
            matches = re.finditer(pattern, job_description.lower())
            skills.update(match.group(0) for match in matches)
        
        return list(skills)
    
    def calculate_skill_match(self, resume_skills: List[str], job_skills: List[str]) -> Tuple[float, List[str], List[str]]:
        """Calculate skill match percentage and identify matched/missing skills."""
        resume_skills_set = set(skill.lower().strip() for skill in resume_skills)
        job_skills_set = set(skill.lower().strip() for skill in job_skills)
        matched_skills = list(resume_skills_set.intersection(job_skills_set))
        missing_skills = list(job_skills_set - resume_skills_set)
        match_percentage = len(matched_skills) / len(job_skills_set) * 100 if job_skills_set else 0
        print('==== DEBUG: Skill Matching ====')
        print('Resume skills:', resume_skills_set)
        print('Job skills:', job_skills_set)
        print('Matched:', matched_skills)
        print('Missing:', missing_skills)
        print('Match %:', match_percentage)
        print('===============================')
        return match_percentage, matched_skills, missing_skills
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using TF-IDF and cosine similarity."""
        # Preprocess texts
        text1_processed = self.preprocess_text(text1)
        text2_processed = self.preprocess_text(text2)
        
        # Create TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform([text1_processed, text2_processed])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return similarity * 100  # Convert to percentage
    
    def score_resume(self, resume_data: Dict, job_description: str) -> Dict:
        """Score resume against job description."""
        # Extract required skills from job description
        job_skills = self.extract_skills_from_job_description(job_description)
        
        # Calculate skill match
        skill_match_percentage, matched_skills, missing_skills = self.calculate_skill_match(
            resume_data['skills'], job_skills
        )
        
        # Calculate overall text similarity
        overall_similarity = self.calculate_text_similarity(
            resume_data['raw_text'], job_description
        )
        
        # Calculate final score (weighted average)
        final_score = (skill_match_percentage * 0.6) + (overall_similarity * 0.4)
        
        return {
            'final_score': round(final_score, 2),
            'skill_match_percentage': round(skill_match_percentage, 2),
            'overall_similarity': round(overall_similarity, 2),
            'matched_skills': matched_skills,
            'missing_skills': missing_skills
        } 
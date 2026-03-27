# job_matcher.py
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Reuse skills and cleaning from resume_processor
from resume_processor import DEFAULT_SKILLS, clean_text

COMMON_SKILLS = {s.lower() for s in DEFAULT_SKILLS}

def extract_skills(text: str) -> set:
    """Extract skills using same logic as Person A"""
    cleaned = clean_text(text)
    words = cleaned.split()
    found = {w for w in words if w in COMMON_SKILLS}

    # Support multi-word skills
    text_lower = cleaned
    for skill in COMMON_SKILLS:
        if " " in skill and skill in text_lower:
            found.add(skill)
    return found

def text_to_vector(text: str, model):
    """Convert text to vector using the same method as Person A"""
    cleaned = clean_text(text)
    tokens = cleaned.split()
    vectors = [model.wv[tok] for tok in tokens if tok in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

class JobMatcher:
    def __init__(self, model):
        self.model = model

    def rank_jobs(self, resume_text: str, job_list: list[dict]):
        if not resume_text.strip():
            print("Resume text is empty!")
            return []

        resume_skills = extract_skills(resume_text)
        resume_vector = text_to_vector(resume_text, self.model)

        results = []
        for job in job_list:
            title = job.get('title', 'Untitled')
            desc = job.get('description', '')

            if not desc.strip():
                results.append({'job': title, 'cosine': 0.0, 'final': 0.0, 'boost': 0.0, 'skills_matched': []})
                continue

            job_vector = text_to_vector(desc, self.model)
            cosine = cosine_similarity(resume_vector.reshape(1, -1), job_vector.reshape(1, -1))[0][0]

            job_skills = extract_skills(desc)
            matched = resume_skills.intersection(job_skills)
            boost = len(matched) * 0.15
            final_score = min(1.0, cosine + boost)

            results.append({
                'job': title,
                'cosine': round(float(cosine), 4),
                'final': round(float(final_score), 4),
                'boost': round(boost, 4),
                'skills_matched': list(matched)
            })

        results.sort(key=lambda x: x['final'], reverse=True)
        return results
# resume_processor.py
import logging
import re
import time
import json
import hashlib
from collections import Counter
from pathlib import Path
from functools import wraps

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

# logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("resume_processor")
logger.setLevel(logging.WARNING) 

_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S",
))
logger.addHandler(_console)

_file_handler = logging.FileHandler(LOG_DIR / "resume_processor.log", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
))
logger.addHandler(_file_handler)

# downloads these nltk if not avalible
for _pkg in ("punkt", "punkt_tab", "stopwords", "wordnet"):
    nltk.download(_pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))  
LEMMATIZER = WordNetLemmatizer()  

# skills to find in resumes
DEFAULT_SKILLS = [
    "python", "java", "javascript", "sql", "html", "css", "react",
    "angular", "node", "django", "flask", "tensorflow", "pytorch",
    "machine learning", "deep learning", "nlp", "docker", "kubernetes",
    "aws", "azure", "gcp", "git", "linux", "mongodb", "postgresql",
    "c++", "c#", "typescript", "scala", "hadoop", "spark", "tableau",
    "power bi", "excel", "communication", "leadership", "teamwork",
]

# saves model
MODEL_DIR = Path("models")
W2V_MODEL_PATH = MODEL_DIR / "word2vec_resume.model"
MODEL_META_PATH = MODEL_DIR / "word2vec_meta.json"


def _timed(func):
    # logs how long a function takes
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("[TIMER] %s completed in %.3f s", func.__name__, elapsed)
        return result
    return wrapper


@_timed
def extract_text_from_file(file_path: str) -> str:
    # reads .txt or .pdf and gives back the raw text
    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning("File not found: %s", file_path)
        return ""

    try:
        size = file_path.stat().st_size
    except OSError as e:
        logger.error("Cannot stat file %s: %s", file_path, e)
        return ""
    if size == 0:
        logger.warning("File has 0 bytes (empty): %s", file_path)
        return ""
    if size > 10 * 1024 * 1024:  # skip files bigger than 10 MB
        logger.warning("File too large (%d bytes), skipping: %s", size, file_path)
        return ""

    ext = file_path.suffix.lower()
    try:
        if ext == ".txt":
            text = file_path.read_text(encoding="utf-8", errors="replace")
        elif ext == ".pdf":
            text = _extract_pdf(file_path)
        else:
            logger.warning("Unsupported file type '%s': %s", ext, file_path)
            return ""
    except (OSError, UnicodeDecodeError, ValueError) as e:
        logger.error("Failed to read %s: %s", file_path, e)
        return ""

    if not text.strip():
        logger.warning("File is empty or unreadable: %s", file_path)
        return ""

    logger.info("Extracted %d characters from %s", len(text), file_path.name)
    return text


def _extract_pdf(file_path: Path) -> str:
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed. Falling back to PyPDF2.")
        return _extract_pdf_pypdf(file_path)  # keep old function as fallback

    try:
        text = ""
        with pdfplumber.open(str(file_path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        logger.info("Extracted %d characters using pdfplumber from %s", len(text), file_path.name)
        return text.strip()
    except Exception as e:
        logger.error("pdfplumber failed on %s: %s", file_path, e)
        return _extract_pdf_pypdf(file_path)  # fallback

def clean_text(text: str) -> str:
    # lowercase everything, remove punctuation, get rid of stopwords
    if not text.strip():
        logger.warning("clean_text received empty text.")
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    cleaned = " ".join(words)

    logger.info("clean_text: %d chars -> %d chars", len(text), len(cleaned))
    return cleaned


def _hash_sentences(sentences: list[list[str]]) -> str:
    # hash the data so we know if resumes changed since last run
    h = hashlib.sha256()
    for sent in sentences:
        h.update(" ".join(sent).encode())
    return h.hexdigest()


@_timed
def get_word2vec_model(
    sentences: list[list[str]],
    retrain: bool = False,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
    workers: int = 4,
) -> Word2Vec:
    # loads saved model if data hasnt changed, otherwise trains new one
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data_hash = _hash_sentences(sentences)

    if W2V_MODEL_PATH.exists() and not retrain:
        if MODEL_META_PATH.exists():
            meta = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
            if meta.get("data_hash") == data_hash:
                logger.info("Data unchanged – loading cached Word2Vec model from %s", W2V_MODEL_PATH)
                return Word2Vec.load(str(W2V_MODEL_PATH))
            else:
                logger.info("Training data changed – retraining Word2Vec model.")
        else:
            # Legacy model without meta – load it but warn
            logger.info("Loading cached Word2Vec model (no meta) from %s", W2V_MODEL_PATH)
            return Word2Vec.load(str(W2V_MODEL_PATH))

    logger.info("Training new Word2Vec model (%d sentences) …", len(sentences))
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
    )
    model.save(str(W2V_MODEL_PATH))
    MODEL_META_PATH.write_text(
        json.dumps({"data_hash": data_hash, "vector_size": vector_size}),
        encoding="utf-8",
    )
    logger.info("Word2Vec model saved to %s", W2V_MODEL_PATH)
    return model


def find_similar_words(model: Word2Vec, word: str, topn: int = 5) -> list[tuple[str, float]]:
    # finds words that are close in meaning using word2vec
    if model is None:
        logger.warning("No Word2Vec model available for similarity query.")
        return []
    if word not in model.wv:
        logger.warning("'%s' not in Word2Vec vocabulary.", word)
        return []
    similar = model.wv.most_similar(word, topn=topn)
    logger.info("Top %d words similar to '%s': %s",
                topn, word, ", ".join(w for w, _ in similar))
    return [(w, round(s, 4)) for w, s in similar]


def get_resume_vector(model: Word2Vec, tokens: list[str]) -> list[float]:
    # averages all word vectors to make one vector for the whole resume
    vectors = [model.wv[tok] for tok in tokens if tok in model.wv]
    if not vectors:
        logger.warning("No tokens in Word2Vec vocabulary – returning zero vector.")
        return [0.0] * model.vector_size
    avg = np.mean(vectors, axis=0)
    logger.info("Resume vector computed from %d / %d tokens.", len(vectors), len(tokens))
    return [round(float(v), 6) for v in avg]


@_timed
def count_skills(text: str, skills: list[str] | None = None) -> dict[str, int]:
    # checks how many times each skill appears (uses \b so "sql" wont match "postgresql")
    if skills is None:
        skills = DEFAULT_SKILLS

    text_lower = text.lower()
    freq: dict[str, int] = {}
    for skill in skills:
        count = len(re.findall(r"\b" + re.escape(skill.lower()) + r"\b", text_lower))
        if count > 0:
            freq[skill] = count

    freq = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
    logger.info("Skills found: %d unique skills", len(freq))
    return freq


@_timed
def top_n_important_words(texts: list[str], n: int = 10) -> list[tuple[str, float]]:
    # uses tfidf to find the most important words across all resumes
    if not texts or all(not t.strip() for t in texts):
        logger.warning("No valid texts for TF-IDF computation.")
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    avg_scores = tfidf_matrix.mean(axis=0).A1
    top_indices = avg_scores.argsort()[::-1][:n]

    results = [(feature_names[i], round(avg_scores[i], 4)) for i in top_indices]
    logger.info("Top %d important words computed.", n)
    return results


def process_resumes(file_paths: list[str], retrain_model: bool = False) -> dict:
    # main function - runs everything in order
    pipeline_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("Starting Resume Processor – %d file(s)", len(file_paths))
    logger.info("=" * 60)

    raw_texts: list[str] = []
    all_tokens: list[list[str]] = []
    per_file: list[dict] = []
    file_data: list[dict] = []

    for fp in file_paths:
        logger.info("-" * 40)
        logger.info("Processing: %s", fp)

        # read resume
        text = extract_text_from_file(fp)
        if not text:
            per_file.append({"file": fp, "status": "skipped", "reason": "empty/corrupted"})
            continue

        raw_texts.append(text)

        # clean and tokenize
        cleaned = clean_text(text)
        if cleaned:
            tokens = word_tokenize(cleaned)
            tokens = [LEMMATIZER.lemmatize(tok) for tok in tokens if tok.isalpha()]
        else:
            tokens = []
        all_tokens.append(tokens)

        # extract skills
        skill_freq = count_skills(text)
        skills_list = list(skill_freq.keys())

        file_data.append({
            "file": fp,
            "tokens": tokens,
            "cleaned": cleaned,
            "skills": skills_list,
            "skill_freq": skill_freq,
        })

    # train or load word2vec
    if all_tokens:
        w2v_model = get_word2vec_model(all_tokens, retrain=retrain_model)
        vocab_size = len(w2v_model.wv)
    else:
        logger.warning("No tokens available – skipping Word2Vec training.")
        w2v_model = None
        vocab_size = 0

    # build resume vectors
    for fd in file_data:
        resume_vector = (
            get_resume_vector(w2v_model, fd["tokens"]) if w2v_model else []
        )
        per_file.append({
            "file": fd["file"],
            "status": "ok",
            "skills": fd["skills"],
            "resume_vector": resume_vector,
            "resume_text": fd["cleaned"],
            "skill_freq": fd["skill_freq"],
            "token_count": len(fd["tokens"]),
        })

    # tfidf top words
    top_words = top_n_important_words(raw_texts)

    similar_words: dict[str, list] = {}
    if w2v_model and top_words:
        for word, _ in top_words[:3]:
            similar_words[word] = find_similar_words(w2v_model, word)

    pipeline_elapsed = time.perf_counter() - pipeline_start

    all_skill_freqs = [fd["skill_freq"] for fd in file_data]

    summary = {
        "files_processed": len(raw_texts),
        "files_skipped": len(file_paths) - len(raw_texts),
        "top_10_words": top_words,
        "aggregate_skills": _merge_skill_counts(all_skill_freqs),
        "word2vec_vocab_size": vocab_size,
        "similar_words": similar_words,
        "per_file": per_file,
        "pipeline_time_sec": round(pipeline_elapsed, 3),
    }

    logger.info("=" * 60)
    logger.info("Processing complete in %.3f s. %d file(s) OK, %d skipped.",
                pipeline_elapsed, summary["files_processed"], summary["files_skipped"])
    logger.info("=" * 60)
    return summary


def _merge_skill_counts(counts: list[dict[str, int]]) -> dict[str, int]:
    merged: Counter = Counter()
    for c in counts:
        merged.update(c)
    return dict(merged.most_common())

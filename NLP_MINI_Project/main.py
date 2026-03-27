# main.py
import json
import sys
from pathlib import Path
from datetime import datetime

from resume_processor import process_resumes, W2V_MODEL_PATH
from job_matcher import JobMatcher

RESUME_DIR = Path("resume")
JOBS_FILE = Path("jobs.json")

def collect_resumes(folder: Path) -> list[str]:
    if not folder.is_dir():
        folder.mkdir(parents=True, exist_ok=True)
        return []
    return sorted(
        str(p) for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".txt", ".pdf"}
    )

def load_jobs() -> list[dict]:
    if not JOBS_FILE.exists():
        print(f"Warning: {JOBS_FILE} not found. Using sample jobs internally.")
        return []
    try:
        with open(JOBS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading jobs.json: {e}")
        return []

def print_header(title: str):
    print("\n" + "=" * 85)
    print(f"  {title.center(81)}")
    print("=" * 85)

def main():
    resume_files = collect_resumes(RESUME_DIR)

    if not resume_files:
        print(f"No resumes found in '{RESUME_DIR.resolve()}'. Please add .pdf or .txt files.")
        sys.exit(0)

    print_header("RESUME PROCESSOR")

    # Process Resumes 
    results = process_resumes(resume_files, retrain_model=False)

    print_header("RESUME PROCESSING SUMMARY")
    print(f"Files processed : {results['files_processed']}")
    print(f"Files skipped   : {results['files_skipped']}")
    print(f"Word2Vec vocab  : {results['word2vec_vocab_size']} words")
    print(f"Pipeline time   : {results['pipeline_time_sec']:.3f} seconds")
    print(f"Run at          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\nTop 10 Important Words (TF-IDF):")
    for rank, (word, score) in enumerate(results["top_10_words"], 1):
        print(f"  {rank:2d}. {word:<20} {score:.4f}")

    print("\nAggregate Skill Frequencies:")
    for skill, count in results["aggregate_skills"].items():
        print(f"  {skill:<20} : {count}")

    # Save full results
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull detailed results saved → results.json")

    # Job Matching 
    print_header("JOB MATCHER")

    jobs = load_jobs()

    try:
        from gensim.models import Word2Vec
        model = Word2Vec.load(str(W2V_MODEL_PATH))
        print(f"Loaded Word2Vec model with {len(model.wv)} words\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    matcher = JobMatcher(model)

    # Store best matches for final summary table
    best_matches = []

    for entry in results["per_file"]:
        if entry.get("status") != "ok":
            continue

        filename = Path(entry["file"]).name
        resume_text = entry["resume_text"]

        print(f"Resume → {filename}")
        print("-" * 85)

        rankings = matcher.rank_jobs(resume_text, jobs)

        # Print top 5 jobs for this resume
        for i, r in enumerate(rankings[:5], 1):
            print(f"{i:2d}. {r['job']:<34} | Final Score: {r['final']:.4f}  "
                  f"(Cosine: {r['cosine']:.4f} + {r['boost']:.4f} boost)")
            if r.get('skills_matched'):
                print(f"     Skills: {', '.join(r['skills_matched'])}")
            print()

        # Save best match
        if rankings:
            best = rankings[0]
            best_matches.append({
                "resume": filename,
                "best_job": best["job"],
                "final_score": best["final"],
                "cosine": best["cosine"],
                "boost": best["boost"],
                "skills_matched": best.get("skills_matched", [])
            })

        print()

    # FINAL BEST MATCH SUMMARY TABLE 
    print_header("FINAL BEST MATCH SUMMARY (Across All Resumes)")

    if best_matches:
        print(f"{'Rank':<4} {'Resume':<42} {'Best Job':<34} {'Final Score':<12} {'Boost':<7} Skills Matched")
        print("-" * 115)

        for rank, match in enumerate(best_matches, 1):
            resume_name = match['resume'][:40] + "..." if len(match['resume']) > 40 else match['resume']
            job_name = match['best_job'][:32] + "..." if len(match['best_job']) > 32 else match['best_job']
            skills_str = ", ".join(match["skills_matched"]) if match["skills_matched"] else "None"
            
            print(f"{rank:<4} {resume_name:<42} {job_name:<34} "
                  f"{match['final_score']:<12.4f} {match['boost']:<7.4f} {skills_str}")
    else:
        print("No successful matches found.")

    print("\n" + "=" * 85)
    print("Resume Processing + Job Matching Completed Successfully!")
    print("=" * 85)

    # Save match results to JSON
    match_results = {
        "timestamp": datetime.now().isoformat(),
        "total_resumes_processed": len(best_matches),
        "best_matches": best_matches
    }
    with open("job_match_results.json", "w", encoding="utf-8") as f:
        json.dump(match_results, f, indent=2, ensure_ascii=False)
    print("Match summary saved → job_match_results.json")

if __name__ == "__main__":
    main()
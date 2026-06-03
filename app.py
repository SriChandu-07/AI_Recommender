
import math
import json
from collections import defaultdict
ITEMS = {
    "I001": {"title": "Python ML Course",      "tags": ["python", "machine learning", "AI", "data science"]},
    "I002": {"title": "Deep Learning Bootcamp", "tags": ["deep learning", "neural networks", "AI", "python"]},
    "I003": {"title": "Data Structures DSA",    "tags": ["algorithms", "data structures", "python", "coding"]},
    "I004": {"title": "Web Dev with React",     "tags": ["react", "javascript", "frontend", "web"]},
    "I005": {"title": "SQL & Databases",        "tags": ["sql", "databases", "backend", "data"]},
    "I006": {"title": "Statistics for Data",    "tags": ["statistics", "data science", "math", "probability"]},
    "I007": {"title": "NLP with Transformers",  "tags": ["nlp", "transformers", "AI", "deep learning"]},
    "I008": {"title": "DevOps & Docker",        "tags": ["devops", "docker", "cloud", "backend"]},
    "I009": {"title": "Computer Vision",        "tags": ["computer vision", "deep learning", "AI", "python"]},
    "I010": {"title": "Linear Algebra Basics",  "tags": ["linear algebra", "math", "data science", "statistics"]},
}
USER_HISTORY={
    "U001": {"I001": 5, "I002": 4, "I006": 3},
    "U002": {"I002": 5, "I007": 5, "I009": 4},
    "U003": {"I001": 4, "I003": 4, "I005": 3},
    "U004": {"I004": 5, "I005": 4, "I008": 3},
    "U005": {"I006": 5, "I010": 5, "I001": 3},
}
def get_user_input():
    print("\nEnter Your Preferences\n")
    name = input("  Your Name       : ").strip() or "Guest"
    user_id = input("  User ID (or NEW): ").strip().upper() or "NEW"
    print("\n  Available topics: python, AI, machine learning,")
    print("  deep learning, data science, math, web, sql,")
    print("  devops, nlp, statistics, algorithms\n")
    raw_interests = input("  Your Interests: ").strip()
    interests = [i.strip().lower() for i in raw_interests.split(",") if i.strip()]

    math_pref = input("  Math comfort level? (low / medium / high): ").strip().lower()
    top_n = input("  How many recommendations? (default 3): ").strip()
    top_n = int(top_n) if top_n.isdigit() else 3

    user_data = {
        "name": name,
        "user_id": user_id,
        "interests": interests,
        "math_pref": math_pref,
        "top_n": top_n,
    }
    return user_data
def build_tfidf_vectors(items):
    docs = {iid: " ".join(meta["tags"]) for iid, meta in items.items()}
    all_terms = set(t for doc in docs.values() for t in doc.split())
    N = len(docs)
    tf = {}
    for iid, doc in docs.items():
        words = doc.split()
        tf[iid] = {term: words.count(term) / len(words) for term in words}
    idf = {}
    for term in all_terms:
        df = sum(1 for doc in docs.values() if term in doc.split())
        idf[term] = math.log((N + 1) / (df + 1)) + 1  

    tfidf = {}
    for iid in docs:
        tfidf[iid] = {term: tf[iid].get(term, 0) * idf.get(term, 0)
                      for term in all_terms}

    return tfidf, idf
def query_vector(interests, idf):
    all_terms = set(idf.keys())
    vec = {}
    for term in all_terms:
        tf_val = interests.count(term) / len(interests) if interests else 0
        vec[term] = tf_val * idf.get(term, 0)
    return vec


def cosine_similarity(vec_a, vec_b):
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)
def content_based_scores(user_interests, tfidf_vectors, idf):
    q_vec = query_vector(user_interests, idf)
    scores = {}
    for iid, item_vec in tfidf_vectors.items():
        scores[iid] = round(cosine_similarity(q_vec, item_vec), 4)
    return scores
def binary_overlap_similarity(user_a_items, user_b_items):
    set_a = set(user_a_items.keys())
    set_b = set(user_b_items.keys())
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0
def collaborative_scores(user_id, user_history, all_items):
    if user_id not in user_history:
        return {} 
    current_user = user_history[user_id]
    seen_items = set(current_user.keys())
    similarities = {}
    for uid, history in user_history.items():
        if uid == user_id:
            continue
        sim = binary_overlap_similarity(current_user, history)
        if sim > 0:
            similarities[uid] = sim

    if not similarities:
        return {}
    scores = defaultdict(float)
    weight_sum = defaultdict(float)
    for uid, sim in similarities.items():
        for iid, rating in user_history[uid].items():
            if iid not in seen_items:
                scores[iid] += sim * rating
                weight_sum[iid] += sim
    predicted = {}
    for iid in scores:
        predicted[iid] = round(scores[iid] / weight_sum[iid] / 5.0, 4)  # normalize 0-1
    return predicted
def cold_start_scores(interests, math_pref, all_items):
    math_tags = {
        "high":   ["linear algebra", "statistics", "math", "probability"],
        "medium": ["data science", "machine learning", "algorithms"],
        "low":    ["web", "frontend", "devops", "sql"],
    }
    boost_tags = math_tags.get(math_pref, [])
    scores = {}
    for iid, meta in all_items.items():
        tags = [t.lower() for t in meta["tags"]]
        match = sum(1 for i in interests if any(i in tag for tag in tags))
        boost = sum(1 for bt in boost_tags if bt in tags)
        raw = (match + boost * 0.5) / (len(tags) + 1)
        scores[iid] = round(raw, 4)
    return scores
def ranking_pipeline(cb_scores, collab_scores, cold_scores,
                     is_cold_start, user_id, user_history, all_items):
    print("\n  ⚙️  Running 4-Step Ranking Pipeline...")
    candidates = list(all_items.keys())
    print(f"  Step 1 ✅ Candidates generated : {len(candidates)} items")
    W_CONTENT = 0.50
    W_COLLAB  = 0.35
    W_COLD    = 0.15
    hybrid = {}
    for iid in candidates:
        cb    = cb_scores.get(iid, 0)
        cf    = collab_scores.get(iid, 0)
        cs    = cold_scores.get(iid, 0)

        if is_cold_start:
            score = cb * 0.65 + cs * 0.35
        else:
            score = cb * W_CONTENT + cf * W_COLLAB + cs * W_COLD

        hybrid[iid] = round(score, 4)

    print(f"  Step 2 ✅ Hybrid scores computed (CB={W_CONTENT}, CF={W_COLLAB}, CS={W_COLD})")

    seen = set(user_history.get(user_id, {}).keys())
    filtered = {iid: s for iid, s in hybrid.items() if iid not in seen}
    print(f"  Step 3 ✅ Filtered seen items  : {len(seen)} removed → {len(filtered)} remain")
    ranked = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    print(f"  Step 4 ✅ Final ranking done    : Top items ready\n")

    return ranked
def display_output(user_data, ranked_items, cb_scores, collab_scores,
                   cold_scores, all_items, top_n):
    print(f"   User ID  : {user_data['user_id']}")
    print(f"   Interests: {', '.join(user_data['interests'])}")
    print(f"   Math Pref: {user_data['math_pref']}")
    print()

    for rank, (iid, score) in enumerate(ranked_items[:top_n], 1):
        title = all_items[iid]["title"]
        tags  = ", ".join(all_items[iid]["tags"])
        cb    = cb_scores.get(iid, 0)
        cf    = collab_scores.get(iid, 0)
        cs    = cold_scores.get(iid, 0)

        print(f"  {rank} {title}")
        print(f"       Item ID      : {iid}")
        print(f"       Tags         : {tags}")
        print(f"       Hybrid Score : {score:.4f}")
        print(f"       Content   : {cb:.4f}  (TF-IDF cosine)")
        print(f"       Collab    : {cf:.4f}  (binary overlap CF)")
        print(f"       Cold Start: {cs:.4f}  (interest+math boost)")
        print()
SKILL_PATTERNS = {
    "beginner":     ["python", "sql", "web", "react"],
    "intermediate": ["machine learning", "data science", "algorithms", "devops"],
    "advanced":     ["deep learning", "nlp", "transformers", "computer vision"],
}

def match_skill_pattern(interests):
    scores = {}
    for level, keywords in SKILL_PATTERNS.items():
        match = sum(1 for i in interests if any(i in kw for kw in keywords))
        scores[level] = match

    best = max(scores, key=scores.get)
    print(f"    Skill Pattern Match  : {best.upper()} level detected")
    print(f"     Pattern Scores       : {json.dumps(scores)}")
    return best

def main():
    user_data = get_user_input()

    user_id   = user_data["user_id"]
    interests = user_data["interests"]
    math_pref = user_data["math_pref"]
    top_n     = user_data["top_n"]
    match_skill_pattern(interests)
    is_cold_start = user_id not in USER_HISTORY
    print(f"\n  Cold Start Status       : {'YES' if is_cold_start else 'NO'}")
    tfidf_vectors, idf = build_tfidf_vectors(ITEMS)
    print(f"  TF-IDF Vectors Built    : {len(tfidf_vectors)} items × {len(idf)} terms")
    cb_scores     = content_based_scores(interests, tfidf_vectors, idf)
    collab_scores = collaborative_scores(user_id, USER_HISTORY, ITEMS)
    cold_scores   = cold_start_scores(interests, math_pref, ITEMS)
    ranked = ranking_pipeline(
        cb_scores, collab_scores, cold_scores,
        is_cold_start, user_id, USER_HISTORY, ITEMS
    )
    display_output(user_data, ranked, cb_scores, collab_scores,
                   cold_scores, ITEMS, top_n)


if __name__ == "__main__":
    main()

"""Corporate Heist preliminary-round solution.

Run from a directory containing train.csv, dev.csv, dev_winners.csv and
test.csv:

    python code/main.py

The script writes submission.csv in the current working directory.  It uses
only NumPy, pandas and scikit-learn, fixes every random seed and is designed
for the stated CPU and memory limits.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


SEED = 42
CURRENT_YEAR = 2026


def first_number(value) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else np.nan


def parse_technical(value) -> float:
    """Return a technical-assessment score on a 0-100 scale."""
    number = first_number(value)
    if pd.isna(number):
        return np.nan
    if number <= 1:
        number *= 100
    return float(np.clip(number, 0, 100))


def parse_aptitude(value) -> float:
    """Return an aptitude score on a 0-10 scale."""
    if pd.isna(value):
        return np.nan
    text = str(value).lower()
    number = first_number(text)
    if pd.isna(number):
        return np.nan
    if "%" in text:
        number /= 10
    elif number <= 1:
        number *= 10
    return float(np.clip(number, 0, 10))


def parse_rating(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    named = {
        "outstanding": 5.0,
        "exceeds expectations": 4.0,
        "meets expectations": 3.0,
        "needs improvement": 2.0,
        "unsatisfactory": 1.0,
        "new joiner - not rated": np.nan,
    }
    if text in named:
        return named[text]
    number = first_number(text)
    return float(np.clip(number, 1, 5)) if not pd.isna(number) else np.nan


def parse_experience(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if "fresher" in text:
        return 0.0
    if "<1" in text:
        return 0.5
    number = first_number(text)
    if pd.isna(number):
        return np.nan
    if "month" in text:
        number /= 12
    return float(number)


def parse_boolean(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return 1.0
    if text in {"no", "n", "false", "0"}:
        return 0.0
    return np.nan


def parse_ctc(value) -> float:
    """Convert the deliberately mixed compensation formats to lakh/year."""
    if pd.isna(value):
        return np.nan
    text = str(value).lower().replace(",", "").strip()
    number = first_number(text)
    if pd.isna(number):
        return np.nan
    if "cr" in text or "crore" in text:
        return number * 100
    if "$" in text:
        # Approximate USD to INR lakh.  The precise exchange rate is not a
        # ranking driver; this prevents unit mistakes from dominating.
        return number * 83 / 100000
    if number > 1000:
        return number / 100000
    return number


def parse_notice_days(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if "immediate" in text or "available now" in text:
        return 0.0
    number = first_number(text)
    if pd.isna(number):
        return np.nan
    return number * 30 if "month" in text else number


def parse_last_job_change(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text == "never":
        return 0.0
    number = first_number(text)
    return number if not pd.isna(number) else np.nan


def parse_awards(value) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value).strip().lower()
    if text in {"", "-", "0", "no", "none", "nan"}:
        return 0.0
    number = first_number(text)
    return max(1.0, number) if not pd.isna(number) else 1.0


def parse_contributions(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"", "not tracked", "unknown", "n/a", "na", "-"}:
        return np.nan
    return first_number(text)


def career_duration(value) -> float:
    if pd.isna(value):
        return np.nan
    total = 0.0
    matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*(months?|mo|years?|yrs?|yr|y)\b",
        str(value).lower(),
    )
    for number, unit in matches:
        amount = float(number)
        total += amount / 12 if unit.startswith("mo") else amount
    return total if matches else np.nan


def title_level(value) -> int:
    text = str(value).lower()
    if re.search(r"\b(vp|vice president|chief)\b", text):
        return 8
    if "head" in text:
        return 7
    if "director" in text:
        return 6
    if "manager" in text:
        return 5
    if re.search(r"\b(staff|principal)\b", text):
        return 4
    if "lead" in text:
        return 3
    if re.search(r"\b(senior|sr\.)\b", text):
        return 2
    if re.search(r"\b(junior|associate)\b", text):
        return 1
    if re.search(r"\b(intern|trainee|apprentice)\b", text):
        return 0
    return 1


ROLE_SKILLS = {
    "Data Engineer": ["sql", "airflow", "spark", "kafka", "etl", "hadoop", "pyspark", "warehouse"],
    "Data Scientist": ["python", "statistics", "sklearn", "machine learning", "pandas", "numpy", "a/b"],
    "ML Engineer": ["python", "pytorch", "tensorflow", "mlops", "deployment", "docker", "kubernetes"],
    "Backend Engineer": ["java", "python", "api", "spring", "node", "sql", "microservice", "postgres"],
    "Frontend Engineer": ["javascript", "typescript", "react", "angular", "vue", "html", "css"],
    "Full Stack Engineer": ["javascript", "react", "node", "api", "sql", "html", "css", "java"],
    "Mobile Engineer": ["android", "kotlin", "swift", "flutter", "ios", "react native", "dart"],
    "DevOps / SRE": ["linux", "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "terraform", "ci-cd"],
    "QA Automation Engineer": ["selenium", "cypress", "testing", "pytest", "java", "python", "automation"],
    "Product Analyst": ["sql", "excel", "tableau", "power bi", "statistics", "experimentation", "a/b"],
}


def role_skill_fit(role, skills, title, path) -> float:
    wanted = ROLE_SKILLS.get(str(role), [])
    if not wanted:
        return 0.0
    text = " ".join([str(skills), str(title), str(path)]).lower()
    return sum(term in text for term in wanted) / len(wanted)


NOTE_PHRASES = [
    "excellent system-design",
    "strong ownership",
    "mentored two",
    "shipped a feature",
    "delivered ahead",
    "migration of a legacy",
    "struggled with the debugging",
    "needed frequent guidance",
    "lukewarm about teamwork",
    "missed two sprint commitments",
    "communication in the panel round was unclear",
]


def normalize_institute(value) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    aliases = {
        "i i t delhi": "iit delhi",
        "iit d": "iit delhi",
        "indian institute of technology delhi": "iit delhi",
        "iitb": "iit bombay",
        "indian institute of technology bombay": "iit bombay",
        "indian institute of technology madras": "iit madras",
        "indian institute of technology kanpur": "iit kanpur",
        "indian institute of technology kharagpur": "iit kharagpur",
        "indian institute of technology roorkee": "iit roorkee",
        "indian institute of technology guwahati": "iit guwahati",
        "indian institute of technology hyderabad": "iit hyderabad",
        "birla institute of technology and science pilani": "bits pilani",
        "delhi technological university formerly dce": "dtu",
        "delhi technological university": "dtu",
        "netaji subhas university of technology": "nsit",
        "nsit delhi": "nsit",
        "nsut": "nsit",
    }
    return aliases.get(text, text)


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=data.index)
    x["age"] = pd.to_numeric(data["age"], errors="coerce")
    x["graduation_year"] = pd.to_numeric(data["graduation_year"], errors="coerce")
    x["experience"] = data["total_experience"].map(parse_experience)
    x["technical"] = data["technical_assessment"].map(parse_technical)
    x["aptitude"] = data["aptitude_score"].map(parse_aptitude)
    x["rating"] = data["last_rating"].map(parse_rating)
    x["kpi"] = data["kpi_met"].map(parse_boolean)
    x["overtime"] = data["overtime_history"].map(parse_boolean)
    x["num_employers"] = pd.to_numeric(data["num_employers"], errors="coerce")
    x["trainings"] = pd.to_numeric(data["trainings_last_year"], errors="coerce")
    x["training_hours"] = pd.to_numeric(data["training_hours"], errors="coerce")
    x["last_job_change"] = data["last_job_change"].map(parse_last_job_change)
    x["current_ctc"] = data["current_ctc"].map(parse_ctc)
    x["expected_ctc"] = data["expected_ctc"].map(parse_ctc)
    x["ctc_hike"] = (x["expected_ctc"] / x["current_ctc"] - 1).clip(-1, 10)
    x["notice_days"] = data["notice_period"].map(parse_notice_days)
    x["awards"] = data["awards"].map(parse_awards)
    x["career_duration"] = data["career_path"].map(career_duration)
    x["career_gap"] = (x["career_duration"] - x["experience"]).abs()
    x["years_since_grad"] = CURRENT_YEAR - x["graduation_year"]
    x["age_at_grad"] = x["age"] - x["years_since_grad"]
    x["experience_vs_grad"] = x["experience"] - x["years_since_grad"]
    x["title_level"] = data["current_title"].map(title_level)
    x["skill_count"] = data["skills"].fillna("").map(
        lambda s: len([t for t in re.split(r"[,;|]", str(s)) if t.strip()])
    )
    x["certification_count"] = data["certifications"].fillna("").map(
        lambda s: len([t for t in re.split(r"[;|]", str(s)) if t.strip()])
    )
    notes = data["recruiter_note"].fillna("").astype(str).str.lower()
    x["note_length"] = notes.str.len()
    x["note_sentence_count"] = notes.str.count(r"\.")
    for i, phrase in enumerate(NOTE_PHRASES):
        x[f"note_{i}"] = notes.str.contains(phrase, regex=False).astype(float)
    x["role_skill_fit"] = [
        role_skill_fit(role, skills, title, path)
        for role, skills, title, path in zip(
            data["applied_role"], data["skills"], data["current_title"], data["career_path"]
        )
    ]
    x["profile_missing_count"] = data.isna().sum(axis=1)
    return x.replace([np.inf, -np.inf], np.nan)


def build_matrix(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    test: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numeric = pd.concat(
        [engineer_features(train), engineer_features(dev), engineer_features(test)],
        ignore_index=True,
    )
    categories = pd.concat(
        [train[categorical_columns], dev[categorical_columns], test[categorical_columns]],
        ignore_index=True,
    ).fillna("Unknown").astype(str)
    categories = pd.get_dummies(categories, dummy_na=False, dtype=np.uint8)
    combined = pd.concat([numeric, categories], axis=1)
    combined = combined.apply(pd.to_numeric, errors="coerce")
    medians = combined.iloc[: len(train)].median(numeric_only=True)
    combined = combined.fillna(medians).fillna(0).astype(np.float32)
    n_train, n_dev = len(train), len(dev)
    return (
        combined.iloc[:n_train].to_numpy(),
        combined.iloc[n_train : n_train + n_dev].to_numpy(),
        combined.iloc[n_train + n_dev :].to_numpy(),
    )


def train_model(x_train: np.ndarray, y: np.ndarray) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=2.0,
        random_state=SEED,
    )
    model.fit(x_train, y)
    return model


def hit_count(dev: pd.DataFrame, predictions: np.ndarray, winner_ids: set[str]) -> int:
    order = np.argsort(-predictions)[:150]
    return int(dev.iloc[order]["candidate_id"].isin(winner_ids).sum())


def contribution_bonus(value: float) -> float:
    if pd.isna(value) or value < 10:
        return 0.0
    if value < 20:
        return 2.0
    if value < 30:
        return 5.0
    if value < 50:
        return 10.0
    if value < 100:
        return 14.0
    return 18.0


def canonical_phone(value) -> str:
    digits = re.sub(r"\D", "", str(value))
    return digits[-10:] if len(digits) >= 10 else digits


def canonical_email(value) -> str:
    text = str(value).strip().lower()
    if "@" not in text:
        return text
    local, domain = text.split("@", 1)
    local = re.sub(r"\+.*$", "", local)
    local = re.sub(r"[^a-z0-9]", "", local)
    return f"{local}@{domain}"


class DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def entity_groups(data: pd.DataFrame) -> np.ndarray:
    """Cluster repeated profiles using canonical phone or e-mail."""
    groups = DisjointSet(len(data))
    seen_phone: dict[str, int] = {}
    seen_email: dict[str, int] = {}
    for position, (phone, email) in enumerate(zip(data["phone"], data["email"])):
        phone_key = canonical_phone(phone)
        email_key = canonical_email(email)
        if phone_key and phone_key in seen_phone:
            groups.union(seen_phone[phone_key], position)
        else:
            seen_phone[phone_key] = position
        if email_key and email_key in seen_email:
            groups.union(seen_email[email_key], position)
        else:
            seen_email[email_key] = position
    return np.array([groups.find(i) for i in range(len(data))])


def current_cycle_ranking(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    test: pd.DataFrame,
    quality_prediction: np.ndarray,
    legacy_prediction: np.ndarray,
) -> pd.DataFrame:
    engineered = engineer_features(test)
    historical_institutes = set(
        pd.concat([train["institute"], dev["institute"]]).map(normalize_institute)
    )
    normalized_test_institute = test["institute"].map(normalize_institute)
    unseen_institute = ~normalized_test_institute.isin(historical_institutes)

    # General historical pedigree/location/referral effects were removed after
    # the reorganisation, so the quality model carries most of the weight.
    base = 0.85 * quality_prediction + 0.15 * legacy_prediction
    contributions = test["public_code_contributions"].map(parse_contributions)
    code_bonus = contributions.map(contribution_bonus)

    new_school_bonus = np.where(
        unseen_institute
        & (engineered["technical"] >= 85)
        & (engineered["role_skill_fit"] >= 0.25),
        6.0,
        np.where(
            unseen_institute
            & (engineered["technical"] >= 80)
            & (engineered["role_skill_fit"] >= 0.15),
            3.0,
            0.0,
        ),
    )

    final_score = base + code_bonus.to_numpy() + new_school_bonus

    # Hard current-cycle policies from the debrief.
    notice_failure = engineered["notice_days"] > 60
    executive_inflation = (engineered["title_level"] >= 6) & (engineered["experience"] < 8)
    other_title_inflation = (
        ((engineered["title_level"] == 5) & (engineered["experience"] < 5))
        | ((engineered["title_level"] == 4) & (engineered["experience"] < 4))
        | ((engineered["title_level"] == 3) & (engineered["experience"] < 3))
    )

    exp_after_grad = engineered["experience"] > engineered["years_since_grad"] + 2
    impossible_grad_age = engineered["age_at_grad"] < 16
    impossible_age_exp = engineered["experience"] > engineered["age"] - 15
    extreme_career_gap = engineered["career_gap"] > 8
    tech_rating_conflict = (engineered["technical"] >= 85) & (engineered["rating"] <= 1)
    severe_count = pd.concat(
        [exp_after_grad, impossible_grad_age, impossible_age_exp, extreme_career_gap, tech_rating_conflict],
        axis=1,
    ).sum(axis=1)
    fabricated = impossible_grad_age | impossible_age_exp | (severe_count >= 2)

    final_score = final_score - np.where(other_title_inflation, 12.0, 0.0)
    ineligible = notice_failure | executive_inflation | fabricated
    final_score = np.where(ineligible, -1_000_000.0, final_score)

    ranked = test[["candidate_id"]].copy()
    ranked["score"] = final_score
    ranked["entity"] = entity_groups(test)
    ranked["original_order"] = np.arange(len(test))

    # The ledger consistently uses the earlier file occurrence where a
    # winning person has two recruiter entries.  Retain that deterministic ID.
    first_record = ranked.groupby("entity")["original_order"].transform("min")
    ranked = ranked[ranked["original_order"] == first_record]
    ranked = ranked.sort_values(["score", "original_order"], ascending=[False, True])
    selected = ranked.head(500).copy()
    return pd.DataFrame(
        {
            "rank": np.arange(1, 501),
            "candidate_id": selected["candidate_id"].to_numpy(),
        }
    )


def locate_data_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "train.csv").exists():
        return cwd
    fallback = Path(__file__).resolve().parent.parent
    if (fallback / "train.csv").exists():
        return fallback
    raise FileNotFoundError("Run from a folder containing train.csv, dev.csv and test.csv")


def main() -> None:
    started = time.time()
    root = locate_data_root()
    train = pd.read_csv(root / "train.csv")
    dev = pd.read_csv(root / "dev.csv")
    winners = pd.read_csv(root / "dev_winners.csv")
    test = pd.read_csv(root / "test.csv")
    target = train["post_hire_score"].to_numpy(dtype=float)

    quality_categories = [
        "applied_role",
        "current_title",
        "company_type",
        "currently_enrolled",
        "overtime_history",
    ]
    legacy_categories = quality_categories + [
        "institute",
        "degree",
        "major",
        "current_city",
        "recruitment_channel",
        "company_size",
    ]

    q_train, q_dev, q_test = build_matrix(train, dev, test, quality_categories)
    quality_model = train_model(q_train, target)
    quality_dev = quality_model.predict(q_dev)
    quality_test = quality_model.predict(q_test)
    del q_train, q_dev, q_test

    l_train, l_dev, l_test = build_matrix(train, dev, test, legacy_categories)
    legacy_model = train_model(l_train, target)
    legacy_dev = legacy_model.predict(l_dev)
    legacy_test = legacy_model.predict(l_test)
    del l_train, l_dev, l_test

    winner_ids = set(winners["candidate_id"])
    quality_hits = hit_count(dev, quality_dev, winner_ids)
    legacy_hits = hit_count(dev, legacy_dev, winner_ids)
    historical_blend = 0.4 * quality_dev + 0.6 * legacy_dev
    blend_hits = hit_count(dev, historical_blend, winner_ids)

    submission = current_cycle_ranking(
        train, dev, test, quality_test, legacy_test
    )
    output_path = Path.cwd() / "submission.csv"
    submission.to_csv(output_path, index=False)

    print(f"Quality model Hits@150: {quality_hits}/150")
    print(f"Legacy model Hits@150:  {legacy_hits}/150")
    print(f"Historical blend Hits@150: {blend_hits}/150")
    print(f"Wrote {output_path} with {len(submission)} ranked candidates")
    print(f"Runtime: {time.time() - started:.1f} seconds")


if __name__ == "__main__":
    main()

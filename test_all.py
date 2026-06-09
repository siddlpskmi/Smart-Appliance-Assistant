import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class ApplianceQASystem:
    def __init__(self, kb_path: str = "kb.csv"):
        # Load knowledge base
        self.df = pd.read_csv(kb_path)

        # ----------- INTENT PATTERNS (rule-based) -----------
        self.intent_patterns = {
            "power_on": [
                ["turn", "on"],
                ["turn", "this", "on"],
                ["power", "on"],
                ["switch", "on"],
                ["switch", "this", "on"],
                ["start"],
                ["start", "this"],
                ["enable"],
            ],
            "power_off": [
                ["turn", "off"],
                ["turn", "this", "off"],
                ["power", "off"],
                ["switch", "off"],
                ["switch", "this", "off"],
                ["shut", "down"],
                ["stop", "using"],
            ],
            "usage": [
                ["how", "to", "use"],
                ["how", "do", "i", "use"],
                ["use", "this"],
                ["operate"],
                ["usage"],
                ["how", "to", "work"],
                ["how", "does", "this", "work"],
            ],
            "cleaning": [
                ["clean"],
                ["how", "to", "clean"],
                ["wash"],
                ["descale"],
                ["maintenance"],
            ],
            "safety": [
                ["safe"],
                ["safety"],
                ["danger"],
                ["precaution"],
                ["warning"],
                ["is", "this", "safe"],
            ],
        }

        # ----------- TF-IDF SETUP -----------
        self.df["canonical_question"] = self.df["canonical_question"].fillna("")
        self.vectorizer = TfidfVectorizer()
        self.kb_tfidf = self.vectorizer.fit_transform(self.df["canonical_question"])

        # ----------- MiniLM EMBEDDING SETUP -----------
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.kb_embeddings = self.embed_model.encode(
            self.df["canonical_question"].tolist(),
            normalize_embeddings=True
        )

    # ----------------------------------------------------
    def detect_intent(self, question: str) -> str:
        """Detect intent by matching keyword patterns."""
        q = question.lower()

        for intent, patterns in self.intent_patterns.items():
            for words in patterns:
                if all(word in q for word in words):
                    return intent

        return "usage"  # fallback

    # ----------------------------------------------------
    def answer(self, device: str, question: str) -> str:
        """Return best matched answer using intent + TF-IDF + MiniLM."""
        intent = self.detect_intent(question)

        # Filter KB by device and intent
        filtered = self.df[
            (self.df["device"] == device) &
            (self.df["intent"] == intent)
        ]

        if filtered.empty:
            filtered = self.df[self.df["device"] == device]

        if filtered.empty:
            return "I don't have information for this device yet."

        filtered_indexes = filtered.index.tolist()

        # ---------- TF-IDF similarity ----------
        user_tfidf = self.vectorizer.transform([question])
        kb_subset_tfidf = self.kb_tfidf[filtered_indexes]
        tfidf_scores = cosine_similarity(user_tfidf, kb_subset_tfidf)[0]

        # ---------- MiniLM similarity ----------
        user_emb = self.embed_model.encode([question], normalize_embeddings=True)
        kb_subset_emb = self.kb_embeddings[filtered_indexes]
        emb_scores = cosine_similarity(user_emb, kb_subset_emb)[0]

        # ---------- HYBRID (50% TF-IDF + 50% MiniLM) ----------
        hybrid = 0.5 * tfidf_scores + 0.5 * emb_scores

        best_local_idx = int(np.argmax(hybrid))
        best_global_idx = filtered_indexes[best_local_idx]

        return self.df.loc[best_global_idx]["answer"]

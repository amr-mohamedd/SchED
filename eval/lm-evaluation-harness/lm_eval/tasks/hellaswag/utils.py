import re

import datasets


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)

def process_docs_generative(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        return {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),  # 0–3
        }
    return dataset.map(_process_doc)


def doc_to_text_generative(doc):
    # Build multiple-choice prompt
    letters = ["A", "B", "C", "D"]
    options_str = "\n".join(f"{l}. {c}" for l, c in zip(letters, doc["choices"]))
    return f"{doc['query']}\n{options_str}\nA, B, C or D:\n"

def doc_to_target_generative(doc):
    # Return gold answer as "A"/"B"/"C"/"D"
    return ["A", "B", "C", "D"][doc["gold"]]

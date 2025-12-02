def doc_to_text(doc):
    return f"{doc['sentence']}\nA. {doc['option1']}\nB. {doc['option2']}\nA or B:\n"

def doc_to_target(doc):
    # Return "A" or "B" instead of "1"/"2"
    answer_map = {"1": "A", "2": "B"}
    return answer_map[doc.get("answer", doc.get("label"))]

def doc_to_choice(doc):
    idx = doc["sentence"].index("_")
    return [
        doc["sentence"][:idx] + doc["option1"],
        doc["sentence"][:idx] + doc["option2"],
    ]

def doc_to_text_generative(doc):
    return f"{doc['sentence']}\nA. {doc['option1']}\nB. {doc['option2']}\nA or B:\n"

def doc_to_target_generative(doc):
    ans = str(doc.get("answer", doc.get("label")))
    answer_to_num = {"1": 0, "2": 1}
    correct_idx = answer_to_num[ans]
    options = [doc["option1"], doc["option2"]]
    return ["A", "B"][correct_idx]  # return "A" or "B"

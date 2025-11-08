# ===========================
# Prompt template for different datasets
# ===========================

dataset_prompts_and_instructions = {

    # NARRATIVE_QA
    "narrative_qa": {
        "instruction": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible.",
        "prompt": """Story:
{context}

{instruction}

Question: {question}

Answer: The answer is""",
        "truncation_message": "... [The rest of the story is omitted]\n\n",
    },

    # QASPER
    "qasper": {
        "instruction": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write 'unanswerable'. If the question is a yes/no question, answer 'yes', 'no', or 'unanswerable'.",
        "prompt": """Article:
{context}

{instruction}

Question: {question}

Answer: The answer is""",
        "truncation_message": "... [The rest of the article is omitted]\n\n",
    },

    # QUALITY
    "quality": {
        "instruction": "You are provided a story and a multiple-choice question with 4 possible answers (marked by A, B, C, D). Choose the best answer by writing its corresponding letter (either A, B, C, or D).",
        "prompt": """Story:
{context}

{instruction}

Question and Possible Answers: {question}

Answer: The answer is""",
        "truncation_message": "... [The rest of the story is omitted]\n\n",
    },

    # CNLI
    "cnli": {
        "instruction": "You are given a non-disclosure agreement and a sentence that proposes a hypothesis based on the agreement. Choose whether the hypothesis is entailed by the agreement, contradicted by the agreement, or not mentioned by (neutral to) the agreement. If the hypothesis is entailed by the agreement, write 'Entailment'. If the hypothesis is contradicted by the agreement, write 'Contradiction'. If the hypothesis is not mentioned by the agreement, write 'Not mentioned'.",
        "prompt": """Contract:
{context}

{instruction}

Hypothesis: {question}

Answer: The answer is""",
        "truncation_message": "... [The rest of the contract is omitted]\n\n",
    },

    # COQA
    "coqa": {
        "instruction": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible.",
        "prompt": """Story:
{context}

{instruction}

Question: {question}

Answer: The answer is""",
        "truncation_message": "... [The rest of the story is omitted]\n\n",
    },
}

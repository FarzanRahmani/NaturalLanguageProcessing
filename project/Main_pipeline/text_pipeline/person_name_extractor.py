from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)
from config import Config
from typing import Any, Dict, List, Tuple, Union


class PersonNameExtractor:
    def __init__(self, model_name: str = Config.NER_MODEL):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.nlp_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple"
        )

    def extract(self, text: str) -> List[str]:
        ner_results = self.nlp_pipeline(text)
        person_names = [
            entity['word']
            for entity in ner_results
            if entity['entity_group'] == 'PER'
        ]
        return person_names
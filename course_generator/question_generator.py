"""Génération des questions de recherche par section."""

import json
from .mistral_client import MistralClient


class QuestionGenerator:
    """Génère 2-4 questions de recherche pour chaque section."""

    PROMPT = """Tu rédiges la section "{titre}" d'un cours.
Objectif pédagogique : {objectif}

Pour rédiger cette section de manière exhaustive et précise, quelles sont les 2 à 4 questions de recherche que tu aimerais poser à une base de documents ?

Retourne UNIQUEMENT un JSON array de strings. Exemple : ["Quelle est la définition de X ?", "Quels sont les avantages de Y ?"]"""

    def __init__(self, client: MistralClient) -> None:
        self.client = client

    def generate(self, section: dict) -> list[str]:
        messages = [
            {
                "role": "system",
                "content": "Tu génères des questions de recherche académiques. Retourne uniquement un JSON array.",
            },
            {
                "role": "user",
                "content": self.PROMPT.format(
                    titre=section["titre"], objectif=section["objectif"]
                ),
            },
        ]
        print(f"  [QST] Génération questions pour : {section['titre'][:50]}...")
        raw = self.client.chat(messages, temperature=0.5, force_json=True)
        try:
            questions = json.loads(raw)
            if not isinstance(questions, list):
                raise ValueError("Pas un array")
            questions = [q for q in questions if isinstance(q, str)]
            if len(questions) < 2:
                questions = [
                    f"Qu'est-ce que {section['titre']} ?",
                    f"Quels sont les éléments clés de {section['titre']} ?",
                ]
            print(f"  [QST] {len(questions)} question(s) générée(s)")
            return questions[:4]
        except Exception as e:
            print(f"  [QST] ERREUR : {e}, fallback")
            return [
                f"Qu'est-ce que {section['titre']} ?",
                f"Quels sont les aspects fondamentaux de {section['titre']} ?",
            ]

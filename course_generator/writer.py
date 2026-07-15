"""Redaction du contenu de section par Mistral."""

import re
from .mistral_client import MistralClient


class SectionWriter:
    """Redige une section de cours avec contexte RAG."""

    _PROMPT_TEMPLATE = r"""Tu es un redacteur scientifique expert en Markdown. Tu es specialise dans la SYNTHESE PEDAGOGIQUE. Tu produis des fiches de revision concises, pas des traites exhaustifs.

=== CONTRAINTE ABSOLUE ===
Le cours ENTIER doit faire MAXIMUM {{MAX_PAGES}} pages A4 (police 11pt, interligne 1.5).
Cela represente un TOTAL MAXIMUM de {{MAX_MOTS_TOTAL}} mots pour l'ensemble du cours.
Pour cette section specifique, ton budget est de EXACTEMENT {{LIMITE_MOTS}} mots.
REGLE ABSOLUE : si tu depasses ce budget, tu as echoue. Recommence en supprimant.

=== RAPPEL DENSITE ===
1 page A4 en Markdown 11pt = environ 250 mots de contenu dense.
Tu dois donc etre EXTREMEMENT CONCIS. Pas de remplissage.
Chaque phrase doit apporter une information nouvelle.

=== METADONNEES ===
Titre : {{TITRE}}
Objectif : {{OBJECTIF}}
Niveau : {{NIVEAU_TEXTE}}

=== CONTEXTE SOURCE (resume a l'essentiel) ===
{{CONTEXTE}}

=== INSTRUCTIONS DE REDACTION ===
1. Style : telegraphique, direct, sans verbiage. Pas d'introductions du type "Depuis quelques annees...".
2. Commence DIRECTEMENT par le concept cle.
3. Paragraphes de 2 a 4 lignes maximum.
4. UN SEUL exemple concret par concept, le plus illustratif. Si tu en as mis 2, supprime le moins bon.
5. Listes (- item) UNIQUEMENT si elles remplacent 3 paragraphes. Sinon, prose compacte.
6. Tableaux UNIQUEMENT pour une comparaison essentielle (max 4 lignes).
7. Code (bloc ```) UNIQUEMENT si indispensable. Sinon, description en 1 ligne.
8. Pas de conclusion de section ("En resume...", "Nous avons vu..."). Stoppe net.
9. Mots-cles en **mot** (premiere occurrence seulement).
10. Emphases en *phrase*.
11. Formules en LaTeX inline ($...$) ou display ($$...$$) seulement si necessaires.
12. Si le contexte est insuffisant : ecris [A completer] sur une ligne, puis stoppe.

=== STRUCTURE Markdown ===
- Niveau 1 : # Titre
- Niveau 2 : ## Titre
- Niveau 3 : ### Titre
- Listes : - item
- Tableaux : | Col1 | Col2 | avec |---|---|
- Code : ```language
- Ancre : {#{{LABEL_SAFE}}}

=== INTERDICTIONS STRICTES ===
- AUCUN LaTeX structurel (\section, \subsection, \begin, \end, \textbf, \textit)
- AUCUN texte hors Markdown (pas de metacommentaire)
- AUCUNE redite avec les sections precedentes
- AUCUNE enumeration de plus de 3 items
- AUCUN preambule HTML ou LaTeX
- ATTENTION : chaque ** doit etre ferme par **. Meme chose pour * et tout bloc ```.

=== RAPPEL FINAL ===
Cours entier : MAXIMUM {{MAX_PAGES}} pages (soit {{MAX_MOTS_TOTAL}} mots).
Cette section : EXACTEMENT {{LIMITE_MOTS}} mots.
Compte tes mots. Si tu depasses, supprime des phrases entieres et recommence.
Redige maintenant."""

    _SYSTEM_PROMPT = (
        "Tu es un redacteur scientifique expert en Markdown. "
        "Tu produis du Markdown pur, inserable dans un document de cours. "
        "Tu es specialise dans la SYNTHESE : tu sais faire passer l'essentiel en peu de mots. "
        "REGLE ABSOLUE : tu respectes TOUJOURS la limite de mots. "
        "Si tu la depasses, tu recommences en reduisant."
    )

    _LIMITE_MOTS = {
        1: 800,
        2: 500,
        3: 300,
    }

    _NIVEAU_TEXT = {
        1: "Chapitre (# - vue d'ensemble, fondations, 800 mots max)",
        2: "Section (## - concepts cles, definitions, 500 mots max)",
        3: "Sous-section (### - detail technique, 300 mots max)",
    }

    _POIDS_NIVEAU = {1: 8, 2: 5, 3: 3}

    # Ratio reel observe en Markdown (titres, listes, espaces prennent de la place)
    _MOTS_PAR_PAGE = 250

    def __init__(
        self,
        client: MistralClient,
        max_pages: float | None = None,
        sections_par_niveau: dict | None = None,
        max_retries: int = 2,
    ) -> None:
        self.client = client
        self.max_retries = max_retries
        self._limite_mots_dynamique = None
        self._max_pages = max_pages
        if max_pages and sections_par_niveau:
            self._limite_mots_dynamique = self._calculer_limites(
                max_pages, sections_par_niveau
            )

    @classmethod
    def _calculer_limites(cls, max_pages: float, sections_par_niveau: dict) -> dict:
        """Repartit un budget total de pages entre niveaux de section."""
        budget_total_mots = max_pages * cls._MOTS_PAR_PAGE
        poids_total = sum(
            cls._POIDS_NIVEAU.get(n, 5) * sections_par_niveau.get(n, 0)
            for n in cls._POIDS_NIVEAU
        )
        if poids_total <= 0:
            return dict(cls._LIMITE_MOTS)
        unite = budget_total_mots / poids_total
        return {
            n: max(80, round(unite * cls._POIDS_NIVEAU[n]))
            for n in cls._POIDS_NIVEAU
        }

    def write(self, section: dict, context: str) -> str:
        niveau = section.get("niveau", 2)
        limites = self._limite_mots_dynamique or self._LIMITE_MOTS
        limite_mots = limites.get(niveau, 500)
        niveau_texte = self._NIVEAU_TEXT.get(niveau, "Section (500 mots max)")

        safe_label = self._make_safe_label(section.get("titre", "section"))
        titre_escaped = self._escape_markdown(section.get("titre", ""))
        objectif_escaped = self._escape_markdown(section.get("objectif", ""))
        contexte_escaped = self._escape_markdown(context)

        max_pages = self._max_pages or 20
        max_mots_total = round(max_pages * self._MOTS_PAR_PAGE)

        prompt = self._PROMPT_TEMPLATE
        prompt = prompt.replace("{{LIMITE_MOTS}}", str(limite_mots))
        prompt = prompt.replace("{{MAX_PAGES}}", str(max_pages))
        prompt = prompt.replace("{{MAX_MOTS_TOTAL}}", str(max_mots_total))
        prompt = prompt.replace("{{TITRE}}", titre_escaped)
        prompt = prompt.replace("{{OBJECTIF}}", objectif_escaped)
        prompt = prompt.replace("{{NIVEAU_TEXTE}}", niveau_texte)
        prompt = prompt.replace("{{CONTEXTE}}", contexte_escaped)
        prompt = prompt.replace("{{LABEL_SAFE}}", safe_label)

        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        print(
            f"  [RED] Redaction : {section.get('titre', '')[:50]}... "
            f"(limite: {limite_mots} mots, budget global: {max_mots_total} mots)"
        )
        content = self._generate_with_retry(messages, limite_mots, max_pages, max_mots_total)
        nb_mots_final = self._compter_mots(content)
        print(f"  [RED] OK ({len(content)} caracteres, ~{nb_mots_final} mots)")

        content = self._sanitize_markdown(content)
        content = self._finaliser(content, limite_mots)
        return content

    def _generate_with_retry(self, messages: list, limite_mots: int, max_pages: float, max_mots_total: int) -> str:
        max_tokens = self._tokens_pour_limite(limite_mots)
        conversation = list(messages)
        content = ""
        for tentative in range(self.max_retries + 1):
            content = self._appel_client(conversation, max_tokens)
            nb_mots = self._compter_mots(content)
            # Tolérance 0% : le budget est strict
            if nb_mots <= limite_mots:
                return content
            if tentative == self.max_retries:
                print(
                    f"  [RED] ATTENTION : {nb_mots} mots pour limite "
                    f"{limite_mots} apres {self.max_retries} tentatives "
                    f"-> troncature de secours"
                )
                return content
            print(
                f"  [RED] Depassement ({nb_mots}/{limite_mots} mots), "
                f"nouvelle tentative {tentative + 1}/{self.max_retries}..."
            )
            conversation = conversation + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        f"ECHEC : ta reponse fait {nb_mots} mots, la limite ABSOLUE "
                        f"est de {limite_mots} mots. Le cours entier doit faire "
                        f"maximum {max_pages} pages (soit {max_mots_total} mots). "
                        f"Supprime IMMEDIATEMENT 2 phrases entieres, compresse "
                        f"les exemples, retire toute redondance. Vise "
                        f"{int(limite_mots * 0.8)} mots. Redige uniquement en "
                        f"Markdown pur, sans commentaire."
                    ),
                },
            ]
        return content

    def _appel_client(self, messages: list, max_tokens: int) -> str:
        try:
            return self.client.chat(messages, temperature=0.2, max_tokens=max_tokens)
        except TypeError:
            return self.client.chat(messages, temperature=0.2)

    @staticmethod
    def _tokens_pour_limite(limite_mots: int) -> int:
        return min(4000, max(400, int(limite_mots * 2.2)))

    @staticmethod
    def _compter_mots(content: str) -> int:
        """Compte les mots bruts (tout caractere separe par whitespace)."""
        if not content:
            return 0
        return len(content.split())

    def _finaliser(self, content: str, limite_mots: int) -> str:
        if self._compter_mots(content) > int(limite_mots * 1.2):
            content = self._tronquer(content, limite_mots)
        content = self._equilibrer_markdown(content)
        return content

    @staticmethod
    def _tronquer(content: str, limite_mots: int) -> str:
        """Coupe au dernier paragraphe entier avant la limite."""
        limite = int(limite_mots * 1.05)
        paragraphes = content.split("\n\n")
        retenus = []
        total = 0
        for p in paragraphes:
            n = len(p.split())
            if retenus and total + n > limite:
                break
            retenus.append(p)
            total += n
        tronque = "\n\n".join(retenus)
        if len(retenus) < len(paragraphes):
            tronque += (
                "\n<!-- Section tronquee automatiquement : depassement de la "
                "limite de mots -->"
            )
        return tronque

    @staticmethod
    def _equilibrer_blocs_code(content: str) -> str:
        backticks = content.count("```")
        if backticks % 2 != 0:
            content += "\n```"
        return content

    @staticmethod
    def _equilibrer_emphasis(content: str) -> str:
        gras_ouvrants = content.count("**") % 2
        if gras_ouvrants:
            content += "**"
        italique_ouvrants = 0
        for m in re.finditer(r'(?<!\*)\*(?!\*)(?! )', content):
            italique_ouvrants += 1
        if italique_ouvrants % 2 != 0:
            content += "*"
        return content

    @classmethod
    def _equilibrer_markdown(cls, content: str) -> str:
        content = cls._equilibrer_blocs_code(content)
        content = cls._equilibrer_emphasis(content)
        return content

    @staticmethod
    def _make_safe_label(titre: str) -> str:
        label = titre.lower()
        for ch in [" ", ":", "\\", "{", "}", "/", "&", "%", "$", "#", "_", "~", "^", "*", "`"]:
            label = label.replace(ch, "-")
        label = label.encode("ascii", "ignore").decode("ascii")
        label = re.sub(r"-+", "-", label).strip("-")[:30]
        if not label:
            label = f"sec-{abs(hash(titre)) % 10000}"
        return label

    @staticmethod
    def _escape_markdown(text: str) -> str:
        if not text:
            return text
        placeholder = "\uE000BACKTICK\uE000"
        text = text.replace("`", placeholder)
        replacements = [
            ("&", "&amp;"),
            ("<", "&lt;"),
            (">", "&gt;"),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        text = text.replace(placeholder, "\\`")
        return text

    @staticmethod
    def _sanitize_markdown(content: str) -> str:
        content = re.sub(r"```markdown\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"```\s*$", "", content, flags=re.MULTILINE)
        content = re.sub(r"```\s*\n", "\n", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content.strip()
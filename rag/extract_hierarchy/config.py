from __future__ import annotations

import re

from docling_core.types.doc import DocItemLabel

# ── Détection des en-têtes de sommaire ──
TOC_HEADING_PATTERNS = [
    r"^sommaire$",
    r"^table des matieres$",
    r"^table des matiere$",
    r"^plan du cours$",
    r"^plan du document$",
    r"^plan de cours$",
    r"^table of contents$",
    r"^contents$",
]

# ── Niveaux hiérarchiques par mot-clé ──
KEYWORD_LEVELS = {
    "chapitre": 1,
    "chapter": 1,
    "partie": 1,
    "part": 1,
    "unite": 1,
    "unit": 1,
    "module": 1,
    "section": 2,
    "sous-section": 3,
    "sous section": 3,
}

LEVEL_LABELS = {1: "chapitre", 2: "titre", 3: "sous_titre"}

# ── Labels Docling ──
LABELS_HEADING = {DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}
LABELS_BODY = {
    DocItemLabel.PARAGRAPH,
    DocItemLabel.TEXT,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CAPTION,
    DocItemLabel.CODE,
    DocItemLabel.FORMULA,
    DocItemLabel.CHECKBOX_UNSELECTED,
    DocItemLabel.CHECKBOX_SELECTED,
}

# Labels à rejeter immédiatement (bruit structurel)
LABELS_NOISE = {
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
    DocItemLabel.FOOTNOTE,
}

# ── Regex d'extraction TOC ──
RE_NUMBERED_WITH_PAGE = re.compile(
    r"^(?P<num>\d+(?:\.\d+)*)[\.\)]?\s+(?P<title>.+?)\s*(?:[\.\-_ ]{2,}|\t)\s*(?P<page>\d{1,4})\s*$"
)

RE_NUMBERED_NO_PAGE = re.compile(
    r"^(?P<num>\d+(?:\.\d+)*)[\.\)]?\s+(?P<title>.+)$"
)

RE_KEYWORD = re.compile(
    r"^(?P<kw>chapitre|chapter|partie|part|unite|unit|module|section)\s+"
    r"(?P<num>[\divxlc]+)\s*[:\-\.]?\s*(?P<title>.+?)"
    r"(?:\s*[\.\-_ ]{2,}\s*(?P<page>\d{1,4}))?\s*$",
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════
#  NOUVEAU : Regex de détection du bruit structurel
# ═══════════════════════════════════════════════════════════

# Téléphones marocains
RE_PHONE_MA = re.compile(
    r"(?:\+212|0[5-7])\s?[\d\s]{8,12}|"
    r"t[eé]l[\s\.]*[:\.]?\s*\(?\+?212\)?[\s\d\-]{8,15}|"
    r"fax[\s\.]*[:\.]?\s*\(?\+?212\)?[\s\d\-]{8,15}",
    re.IGNORECASE,
)

# Emails institutionnels
RE_EMAIL = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)

# Bibliographie : et al, années, pp., Vol., DOI, URLs
RE_BIBLIO = re.compile(
    r"(?:et\s+al[\.\,]|"
    r"20\d{2}[a-z]?[;\.\,]|"
    r"pp?\.\s*\d+|"
    r"Vol\.\s*\d+|"
    r"DOI[\s\:]*|"
    r"https?://|"
    r"www\.[a-zA-Z0-9.-]+|"
    r"[A-Z][a-z]+\s+[A-Z]\.[A-Z]?\.?\s*,?\s*20\d{2})",
    re.IGNORECASE,
)

# Lignes de points (formulaires vides)
RE_FORM_DOTS = re.compile(r"^[\.\s\u2026\u00B7\-]{10,}$")

# Formulaires génériques
RE_FORM_GENERIC = re.compile(
    r"\b(?:MOD\s*E\s*LE|DATE\s*DE\s*FABRICATION|"
    r"CAHIER\s*DES\s*CHARGES|GUIDE|RAPPORT\s*NATIONAL|"
    r"MINIST[ÈE]RE|UNIVERSIT[ÉE]|FACULT[ÉE]|"
    r"DIRECTION|DIVISION|SERVICE\s+\w+)\b",
    re.IGNORECASE,
)

# Mots-clés de pages de garde / poubelle
GARBAGE_HEADING_KEYWORDS = {
    "cahier des charges", "guide", "rapport", "rapport national",
    "ministère", "ministere", "université", "universite",
    "faculté", "faculte", "direction", "division",
    "modèle", "modele", "date de fabrication",
}

# Mots-clés médicaux pour valider un heading
MEDICAL_KEYWORDS = {
    "diabète", "diabete", "vih", "sida", "hépatite", "hepatite",
    "dépistage", "depistage", "traitement", "thérapeutique",
    "therapeutique", "vaccin", "prévention", "prevention",
    "épidémiologie", "epidemiologie", "pathologie", "clinique",
    "biologie", "chimiothérapie", "chimiotherapie", "radiologie",
    "oncologie", "cardiologie", "pneumologie", "gastro",
    "nephrologie", "urologie", "gynécologie", "gynecologie",
    "pédiatrie", "pediatrie", "obstétrique", "obstetrique",
    "infectiologie", "bactériologie", "bacteriologie", "virologie",
    "parasitologie", "mycologie", "immunologie", "hémato",
    "hemato", "transfusion", "pharmacologie", "toxicologie",
    "anesthésie", "anesthesie", "réanimation", "reanimation",
    "chirurgie", "orthopédie", "orthopedie", "dermatologie",
    "ophtalmologie", "orl", "rhumatologie", "endocrinologie",
    "neurologie", "psychiatrie", "gériatrie", "geriatrie",
    "nutrition", "diététique", "dietetique", "hygiène", "hygiene",
    "stomatologie", "odontologie", "médecine", "medecine",
    "soins", "urgences", "samu", "smur", "hôpital", "hopital",
    "clinique", "laboratoire", "analyse", "biomarqueur",
    "métastase", "metastase", "tumeur", "cancer", "carcinome",
    "sarcome", "lymphome", "leucémie", "leucemie", "métastase",
    "insuffisance", "déficit", "deficit", "syndrome", "symptôme",
    "symptome", "diagnostic", "pronostic", "évolution", "evolution",
    "complication", "surveillance", "suivi", "dose", "posologie",
    "indication", "contre-indication", "contre indication",
    "effet indésirable", "effet indesirable", "toxicité", "toxicite",
    "résistance", "resistance", "sensibilité", "sensibilite",
    "spectre", "bactérie", "bacterie", "virus", "parasite",
    "champignon", "protozoaire", "helminth", "arthropode",
    "vector", "vecteur", "transmission", "prévalence", "prevalence",
    "incidence", "morbidité", "morbidite", "mortalité", "mortalite",
    "espérance", "esperance", "qualité de vie", "qualite de vie",
    "charge de morbidité", "DALY", "QALY", "coût", "cout",
    "efficacité", "efficacite", "efficience", "recommandation",
    "consensus", "protocole", "algorithme", "score", "échelle",
    "echelle", "classification", "stade", "grade", "seuil",
    "critère", "critere", "inclusion", "exclusion", "randomisation",
    "essai clinique", "cohorte", "cas-témoin", "cas temoin",
    "métanalyse", "meta analyse", "meta-analyse", "revue systématique",
    "revue systematique", "guide", "recommandation", "HAS",
    "OMS", "WHO", "CDC", "INVS", "SPF", "ANSM", "FDA",
    "EMA", "AFSSAPS", "ANSES", "INCa", "INCA",
    "cirrhose", "fibrose", "nécrose", "necrose", "inflammation",
    "cytolyse", "cholestase", "ictère", "ictere", "ascite",
    "encéphalopathie", "encephalopathie", "hypertension",
    "portale", "varice", "hémorragie", "hemorragie", "transplantation",
    "greffe", "dialyse", "hémodialyse", "hemodialyse", "épuration",
    "epuration", "insuffisance rénale", "insuffisance renale",
    "protéinurie", "proteinurie", "hématurie", "hematurie",
    "glycosurie", "acidocétose", "acidocetose", "hypoglycémie",
    "hypoglycemie", "hyperglycémie", "hyperglycemie", "rétinopathie",
    "retinopathie", "neuropathie", "nephropathie", "nephropathie",
    "angiopathie", "pied diabétique", "pied diabetique", "ulcère",
    "ulcere", "gangrène", "gangrene", "amputation", "décompensation",
    "decompensation", "coma", "acidose", "alcalose", "déshydratation",
    "deshydratation", "dénutrition", "denutrition", "obésité", "obesite",
    "surpoids", "dyslipidémie", "dyslipidemie", "athérosclérose",
    "atherosclerose", "infarctus", "angor", "arythmie", "fibrillation",
    "flutter", "tachycardie", "bradycardie", "bloc", "hypertrophie",
    "dilatation", "insuffisance cardiaque", "œdème", "oedeme",
    "embolie", "thrombose", "phlébite", "phlebite", "dissection",
    "anévrisme", "anevrisme", "endocardite", "péricardite",
    "pericardite", "myocardite", "cardiomyopathie", "valvulopathie",
    "sténose", "stenose", "régurgitation", "regurgitation", "prothèse",
    "prothese", "pacemaker", "défibrillateur", "defibrillateur",
    "resynchronisation", "ablation", "catheterisme", "coronarographie",
    "scintigraphie", "IRM", "TDM", "scanner", "échographie",
    "echographie", "échocardiographie", "echocardiographie",
    "holter", "épreuve", "epreuve", "effort", "spirométrie",
    "spirometrie", "plethysmographie", "diffusion", "capacité",
    "capacite", "volume", "débit", "debit", "PaO2", "PaCO2",
    "SaO2", "FiO2", "CPAP", "VNI", "VM", "ECMO", "oxygénation",
    "oxygenation", "bronchodilatateur", "corticoïde", "corticoide",
    "antibiotique", "antiviral", "antifongique", "antiparasitaire",
    "antituberculeux", "antirétroviral", "antiretroviral", "interféron",
    "interferon", "ribavirine", "sofosbuvir", "daclatasvir",
    "velpatasvir", "glecaprevir", "pibrentasvir", "voxilaprevir",
    "metformine", "sulfamide", "glinide", "iDPP4", "GLP1", "SGLT2",
    "insuline", "basale", "bolus", "prandiale", "analogue", "dégludec",
    "degludec", "glargine", "détémir", "detemir", "lispro", "aspart",
    "glulisine", "NPH", "prémélangée", "premelangee", "pompe",
    "stylo", "aiguille", "glycémie", "glycemie", "HbA1c", "fructosamine",
    "microalbuminurie", "créatinine", "creatinine", "DFG", "MDRD",
    "CKD-EPI", "urée", "uree", "ionogramme", "bilan hépatique",
    "bilirubine", "transaminases", "ASAT", "ALAT", "GGT", "PAL",
    "TP", "INR", "facteur", "albumine", "protides", "préalbumine",
    "prealbumine", "ammoniémie", "ammoniemie", "lactates", "lactate",
    "pH", "bicarbonates", "base", "exces", "anion", "osmolarité",
    "osmolarite", "natrémie", "natremie", "kaliémie", "kaliemie",
    "chlorémie", "chloremie", "calcémie", "calcemie", "phosphorémie",
    "phosphoremie", "magnésémie", "magnesemie", "ferrémie", "ferremie",
    "ferritine", "transferrine", "saturation", "CRP", "PCT",
    "VS", "fibrinogène", "fibrinogene", "D-dimères", "D-dimeres",
}

# Seuils
NOISE_CHAR_RATIO_THRESHOLD = 0.35   # > 35% non-alpha = bruit
GARBAGE_HEADING_MIN_WORDS = 12      # headings < 12 mots suspects
MIN_CHUNK_WORDS = 30                # rejet final si < 30 mots
CHUNK_MERGE_WORDS = 50              # fusion si < 50 mots
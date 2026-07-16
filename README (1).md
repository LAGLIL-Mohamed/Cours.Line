# Architecture Web RAG — Workspace Isolé

## 📁 Structure des fichiers

```
Projet python - RAG/          ← Projet original (INTOUCHÉ)
├── main.py                   ← Ton main interactif original
├── main_batch.py             ← ✅ NOUVEAU : version batch non-interactive
├── reterevel.py
├── rag/
│   ├── vectorization/
│   └── course_generator/
└── ...

web_sessions/                 ← Workspaces temporaires (auto-créés)
├── session_20250716_143022_a1b2c3d4/
│   ├── main_batch.py
│   ├── reterevel.py
│   └── rag/
│       ├── data/
│       │   ├── input/        ← PDFs injectés
│       │   ├── output/       ← Résultats générés
│       │   └── vector_store/   ← Index FAISS regénéré
│       └── ...
└── session_20250716_143045_e5f6g7h8/
    └── ...

Interface web (dossier à part) :
├── app.py                    ← ✅ FastAPI + formulaire HTML
├── workspace_manager.py      ← ✅ Orchestrateur de workspaces
└── requirements.txt
```

---

## 🚀 Installation

### 1. Prérequis

- Python 3.10+
- Le projet RAG original fonctionnel (avec `main.py`, `reterevel.py`, etc.)
- Clé API Mistral configurée

### 2. Copier `main_batch.py` dans le projet original

```bash
cp main_batch.py "C:\Users\dell\Desktop\Projet python - RAG\"
```

**Pourquoi ?** Le workspace manager copie le projet original entier. Si `main_batch.py` n'est pas dans le projet original, il ne sera pas copié dans les workspaces.

### 3. Installer les dépendances web

```bash
pip install fastapi uvicorn python-multipart
```

### 4. Configurer les variables d'environnement

```bash
# Windows
set RAG_ORIGINAL_PROJECT=C:\Users\dell\Desktop\Projet python - RAG
set MISTRAL_API_KEY=ta_cle_api_ici

# Linux/Mac
export RAG_ORIGINAL_PROJECT=/home/user/Projet\ python\ -\ RAG
export MISTRAL_API_KEY=ta_cle_api_ici
```

---

## 🎓 Utilisation

### Mode 1 : Interface Web (recommandé)

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Ouvrir `http://localhost:8000` dans le navigateur :
1. Uploader les PDFs
2. Choisir le nombre de pages et chapitres
3. Cocher/décocher "Générer un TD"
4. Cliquer sur **Lancer la génération**
5. Télécharger le ZIP avec `course.md`, `course.json`, `plan.json`, `td.md`

### Mode 2 : Ligne de commande (test/debug)

```bash
python workspace_manager.py \
    --project "C:\Users\dell\Desktop\Projet python - RAG" \
    --pdfs doc1.pdf doc2.pdf \
    --pages 10 \
    --chapters 3 \
    --keep  # Garde le workspace pour inspection
```

### Mode 3 : Script Python direct

```python
from workspace_manager import WorkspaceManager
from pathlib import Path

manager = WorkspaceManager(Path(r"C:\Users\dell\Desktop\Projet python - RAG"))

config = {
    "max_pages": 10,
    "num_chapters": 3,
    "generate_td": True,
    "skip_extraction": False,
    "skip_summary": False,
    "skip_indexing": False,
    "interactive_plan": False,
}

result = manager.run_session(
    pdf_files=[Path("doc1.pdf"), Path("doc2.pdf")],
    config=config,
    keep_workspace=False
)

# Récupérer les fichiers
for fname, fbytes in result["files"].items():
    with open(fname, "wb") as f:
        f.write(fbytes)
```

---

## ⚙️ Configuration du pipeline (dict `config`)

| Clé | Type | Défaut | Description |
|-----|------|--------|-------------|
| `max_pages` | int | 10 | Nombre de pages du cours |
| `num_chapters` | int | 3 | Nombre de chapitres |
| `chapters` | list | [] | Structure chapitres (vide = auto) |
| `skip_extraction` | bool | False | Sauter phase 1 si chunks existent |
| `skip_summary` | bool | False | Sauter phase 2 si résumé existe |
| `skip_indexing` | bool | False | Sauter phase 2bis si index existe |
| `interactive_plan` | bool | False | Activer modifications du plan (web: toujours False) |
| `plan_modifications` | list | [] | Instructions de modification du plan |
| `pre_validated_plan` | dict | None | Plan JSON pré-validé (saute la génération) |
| `generate_td` | bool | True | Générer le TD de révision |
| `mistral_api_key` | str | env | Clé API Mistral |

---

## 🔒 Sécurité et isolation

- **Projet original jamais modifié** : seules des copies temporaires sont créées
- **Sessions indépendantes** : chaque utilisateur a son propre workspace
- **Nettoyage automatique** : le workspace est supprimé après récupération des résultats
- **Pas de conflit de données** : les anciens PDFs et résultats restent intacts
- **Parallélisme possible** : plusieurs sessions peuvent tourner simultanément

---

## 🐛 Dépannage

### "main_batch.py introuvable dans le workspace"
→ Copiez `main_batch.py` à la racine du projet original.

### "Projet original introuvable"
→ Définissez `RAG_ORIGINAL_PROJECT` avec le chemin absolu correct.

### "Clé API Mistral non définie"
→ Définissez `MISTRAL_API_KEY` dans les variables d'environnement.

### Le pipeline plante en phase 2/3/4
→ Lancez avec `--keep` pour conserver le workspace et inspecter les logs.

### Les chemins dans CourseConfig sont absolus
→ Si `course_generator/config.py` a des chemins hardcodés, vous devrez peut-être le modifier pour qu'il utilise des chemins relatifs ou qu'il lise `RAG_PROJECT_ROOT` depuis l'environnement.

---

## 📄 Fichiers générés

| Fichier | Description |
|---------|-------------|
| `course.md` | Cours complet en Markdown |
| `course.json` | Cours structuré en JSON |
| `plan.json` | Plan de cours validé |
| `td.md` | TD de révision (optionnel) |

<p align="center">
  <img src="interface/static/logo.png" alt="Cours.Line Logo" width="400">
</p>

<h1 align="center">Cours.Line</h1>

<p align="center">
  <strong>Générez automatiquement des cours structurés et des travaux dirigés à partir de vos documents PDF.</strong>
</p>

<p align="center">
  <a href="#-fonctionnalités">Fonctionnalités</a> •
  <a href="#-démarrage-rapide">Démarrage rapide</a> •
  <a href="#-utilisation">Utilisation</a> •
  <a href="#-prérequis">Prérequis</a>
</p>

---

## ✨ Qu'est-ce que Cours.Line ?

**Cours.Line** est une application intelligente qui transforme vos documents numériques (articles scientifiques, supports de cours, polycopiés PDF) en **cours pédagogiques complets** et **travaux dirigés (TD)** prêts à l'emploi.

Que vous soyez enseignant, étudiant ou formateur, Course.Line analyse vos sources, en extrait l'essentiel et génère pour vous :
- 📘 Un **cours structuré** en chapitres et sections
- 📝 Un **TD de révision** avec QCM, questions courtes et exercices pratiques
- 📄 Des documents finaux au format PDF académique

---

## 🚀 Fonctionnalités

| Fonctionnalité | Description |
|----------------|-------------|
| 📥 **Import PDF** | Uploadez un ou plusieurs documents PDF en une seule fois |
| 🧠 **Analyse intelligente** | Extraction automatique de la hiérarchie et des concepts clés |
| 📐 **Plan de cours** | Génération d'un plan pédagogique cohérent et progressif |
| ✍️ **Rédaction automatique** | Rédaction du contenu complet en Markdown structuré |
| 🎯 **Génération de TD** | Création d'exercices variés (QCM, questions, cas pratiques) |
| 📄 **Export PDF** | Conversion académique avec mise en page professionnelle |
| 🌐 **Interface Web** | Utilisation simple via navigateur, sans ligne de commande |

---

## 📋 Prérequis

- **Python 3.11** ou supérieur
- Une **clé API Mistral** (obtenable gratuitement sur [console.mistral.ai](https://console.mistral.ai))
- Vos documents au format **PDF**

---

## 🛠️ Installation

### 1. Cloner le projet

```bash
git clone https://github.com/Mo7hamedd/Cours.Line.git
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer la clé API

Créez un fichier `.env` à la racine du projet :

```env
MISTRAL_API_KEY=votre_clé_api_ici
```

Ou définissez-la directement dans votre terminal :

```bash
# Windows
set MISTRAL_API_KEY=votre_clé_api_ici

# macOS / Linux
export MISTRAL_API_KEY=votre_clé_api_ici
```

---

## 💻 Utilisation

### Mode Web (recommandé)

Lancez l'interface web et ouvrez votre navigateur :


Rendez-vous sur `http://localhost:8000` et suivez les étapes :

1. **Uploader** vos fichiers PDF
2. **Choisir** le nombre de pages et de chapitres souhaités
3. **Lancer** la génération
4. **Télécharger** votre cours et votre TD au format ZIP


## 📦 Structure du projet

```
course-line/
├── interface/          # Application web FastAPI
├── course_generator/   # Moteur de génération de cours
├── rag/                # Pipeline d'analyse et de vectorisation
├── main_batch.py       # Script d'exécution batch
├── requirements.txt    # Dépendances Python
└── README.md           # Ce fichier
```

---

## 🎯 Exemple de sortie

### Cours généré
- Plan structuré (chapitres → sections → objectifs pédagogiques)
- Contenu rédigé en Markdown avec formules LaTeX supportées
- Mise en page académique professionnelle

### TD généré
- 3 questions QCM avec correction
- 2 questions courtes
- 1 exercice pratique / cas d'application
- Corrigé succinct intégré

---




<p align="center">
  <sub>M1 BIBDA @FSJ</sub>
</p>

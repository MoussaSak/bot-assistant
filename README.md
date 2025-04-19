# Assistant a base d'IA
Ce Projet est un exemple qui démontre un simple cas d'usage d'un assistant IA basé sur Ollama et la méthode RAG. Pour faire son boulot il va prendre les fichiers `.pdf` disponible dans le répertoire `context` puis il va faire l'embedding du texte pour construire après une base de connaissance de type `faiss`. Cette base de connaissance véctoriel va aider le Modèle a générer des réponses plus pertinentes.  

## Procédure d'installation de Ollama 
### Prérequis
- Cloner le projet
- Installer [Python](https://www.python.org/downloads/) (toute version au dessus de la 3.10)
- Installer [Ollama](https://ollama.com/download)

### Lancement du projet
- Lancer le service `ollama` via la commande: 
```bash
sudo systemctl start ollama.service
``` 
- Télécharger le modèle utilisé dans notre application avec la commande: 
```bash
sudo ollama pull llama3.2:latest
``` 
P.S.: si vous changer le modèle utilisé dans le code, vous devez faire cette étape pour le modèle concerné.

- Installer les dépendances du projet via la commande (de préférence utiliser un environnement virtuel python [venv](https://docs.python.org/fr/3/tutorial/venv.html)):
```bash
pip install -r rag/requirements.txt
```

- Se placer dans le dossier du projet et lancer la commande :
```
streamlit run rag/app.py
```

- Une fenêtre web va s'ouvrir avec comme URL :
```
http://localhost:8501
```

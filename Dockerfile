# Utiliser une image officielle Python comme base
FROM python:3.12-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application dans le conteneur
COPY . .

# Exposer le port sur lequel l'application va écouter
EXPOSE 8000

# Lancer l'application FastAPI avec Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

###
### docker build -t detection-fraude-fastapi-app .
### docker run -d -p 8000:8000 detection-fraude-fastapi-app
### docker save -o C:\Users\Utilisateur\Music\final\detection-fraude-fastapi-app.tar detection-fraude-fastapi-app
## docker tag <image_name>:<tag> <dockerhub_username>/<repository_name>:<tag>
## docker tag detection-fraude-fastapi-app:latest anicet11/detection-fraude-fastapi-app:v1

## docker push <dockerhub_username>/<repository_name>:<tag>
## docker push anicet11/detection-fraude-fastapi-app:v1

# FastAPI Lexical Resources

## Déploiement sans docker

Requirements: 
- python >= 3.9

### Installation dépendances

```
pip install fastapi uvicorn
pip install git+https://github.com/moreymat/magnitude.git@0.1.143.1#egg=pymagnitude
```

Dans le fichier `fastapi-lexical-resources/api-config.config`, changer la valeur de `deployment_method` en `local` et créez la configuration que vous souhaitez.

Depuis le répertoire `fastapi-lexical-resources/`

```
bash ./start_service.sh
```

## Créer une image docker

Requirements:
- Python >= 3.9
- Docker >= 20.X

Dans le fichier `fastapi-lexical-resources/api-config.config`, changer la valeur de `deployment_method` en `docker` et créez la configuration que vous souhaitez.

Depuis le répertoire `fastapi-lexical-resources/`

```
bash ./start_service.sh
```


## Preprocessing et preloading

Il y a cinq étapes avant de lancer le service qui peuvent prendre un certain temps:

- Prise en comptes du fichier config
- Téléchargement des embeddings manquant
- Conversion de ces derniers    (non sauvegardé localement si fait via docker)
- Préchargement avec la méthode most_similar 
- Préchargement des mots-clés des reférentiels 

## API Documentation

Une fois le service lancé (localement), une documentation intéractible est disponible à cette adresse: http://127.0.0.2:8000/docs#/

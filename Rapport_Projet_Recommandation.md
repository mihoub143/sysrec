# Moteur de Recommandation - Micro-Aventures
**Auteurs :** Zahra Kodia Aouina & Siwar Gorrab  
**Classe :** 2 LNBI - ISG TUNIS  

---

## 1. Problématique
Avec la multiplication des offres de loisirs et de tourisme local, les utilisateurs se retrouvent souvent perdus face à une surabondance d'options (surcharge d'information). L'objectif de ce projet est de concevoir et développer un moteur de recommandation intelligent capable de suggérer des "Micro-Aventures et Expériences Insolites" (ex: Nuit en igloo, Vol en montgolfière) adaptées aux préférences uniques de chaque utilisateur. 

Le défi technique réside dans la nécessité de combiner plusieurs approches de filtrage pour pallier les défauts individuels de chaque méthode (comme le problème du démarrage à froid) et d'offrir une interface explicable et transparente.

## 2. Méthodologie
Notre système repose sur une base de données **PostgreSQL** personnalisée (`sr`) contenant trois tables principales (Produit, Users, Notes). Nous avons développé un modèle hybride combinant trois algorithmes :

### A. Filtrage basé sur le contenu (Content-Based - TD4)
Nous analysons le texte des descriptions des expériences pour recommander des items similaires à ceux que l'utilisateur a déjà bien notés.
- **NLP** : Utilisation de NLTK pour la tokenisation (`word_tokenize`), le stemming (`FrenchStemmer`) et la suppression des mots vides (`stopwords`).
- **Représentation** : Création d'une Matrice Binaire associant chaque produit à ses racines de mots.
- **Similarité** : Calcul de la similarité Cosinus entre les vecteurs des produits.

### B. Filtrage Collaboratif (User-Based - TD5)
Nous exploitons la sagesse de la foule en recommandant ce qu'ont aimé des utilisateurs similaires.
- **Matrice Utilisateur-Item** : Création d'un pivot à partir de l'historique des notes.
- **Similarité** : Calcul de la similarité Cosinus entre les utilisateurs.
- **Prédiction** : Pour un produit non noté, calcul d'une note estimée basée sur la moyenne pondérée des notes des "K Plus Proches Voisins" (KNN).

### C. Recommandation Temporelle (Time-Aware)
Nous intégrons le facteur temps de deux manières :
- **Décroissance Temporelle** : Les notes récentes ont un poids mathématique plus fort (fonction exponentielle) que les notes anciennes.
- **Saisonnalité** : Les activités correspondant à la saison actuelle (ex: "Hiver" pour le Bivouac en raquettes) reçoivent un boost de pertinence.

### D. Modèle Hybride et Explicabilité
Les scores (normalisés) des trois algorithmes sont combinés (moyenne pondérée). Le système détermine quel algorithme a le plus contribué au score final pour afficher une explication claire à l'utilisateur (ex: *"Recommandé car des explorateurs similaires ont adoré"*).

## 3. Résultats et Évaluation (Bonus)
L'application développée en **Streamlit** offre une interface ergonomique et dynamique.

Pour évaluer la pertinence, nous avons implémenté les métriques classiques des systèmes de recommandation :
- **Precision@K** : Mesure la proportion d'expériences pertinentes parmi le Top-K recommandé.
- **Recall@K** : Mesure la capacité du système à retrouver les bonnes expériences pour l'utilisateur.

Nos tests sur le dataset synthétique montrent une excellente capacité du modèle hybride à s'adapter aux profils variés, justifiant pleinement l'intégration des 3 approches. L'interface Streamlit permet une navigation fluide et l'explicabilité renforce la confiance de l'utilisateur envers les recommandations.

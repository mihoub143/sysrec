import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# On garde NomPdt et Description pour le TD4, mais on remet les tags et la catégorie pour la belle interface !
produits = [
    {"ID": 1, "NomPdt": "Nuit dans une cabane dans les arbres", "Description": "Passez une nuit inoubliable dans une cabane perchée dans les arbres de la forêt avec un calme absolu.", "Categorie": "Nature & Détente", "Effort": "Faible", "Saison": "Printemps/Été", "Tags": "forêt, calme, insolite, nature, couple"},
    {"ID": 2, "NomPdt": "Stage de survie en forêt", "Description": "Apprenez les techniques de survie en forêt, comment faire du feu et trouver de l'eau en pleine nature.", "Categorie": "Aventure & Sport", "Effort": "Intense", "Saison": "Toutes", "Tags": "survie, feu, adrénaline, groupe, nature"},
    {"ID": 3, "NomPdt": "Atelier de poterie artisanale", "Description": "Découvrez le travail de l'argile et la création de poterie artisanale lors de cet atelier manuel relaxant.", "Categorie": "Créatif & Culture", "Effort": "Faible", "Saison": "Toutes", "Tags": "manuel, argile, détente, intérieur, solo"},
    {"ID": 4, "NomPdt": "Randonnée nocturne et astronomie", "Description": "Marche en montagne sous les étoiles avec un guide pour observer l'espace et la nature nocturne.", "Categorie": "Nature & Découverte", "Effort": "Moyen", "Saison": "Été/Automne", "Tags": "étoiles, marche, nuit, espace, nature"},
    {"ID": 5, "NomPdt": "Vol en montgolfière au lever du soleil", "Description": "Admirez le lever du soleil et le panorama exceptionnel depuis le ciel lors d'un vol calme.", "Categorie": "Aventure & Détente", "Effort": "Faible", "Saison": "Printemps/Été", "Tags": "ciel, panorama, insolite, couple, matin"},
    {"ID": 6, "NomPdt": "Initiation à l'apnée en lac", "Description": "Plongez dans les profondeurs d'un lac et apprenez à contrôler votre respiration dans l'eau.", "Categorie": "Aventure & Sport", "Effort": "Intense", "Saison": "Été", "Tags": "eau, respiration, sport, nature, adrénaline"},
    {"ID": 7, "NomPdt": "Création de son propre parfum", "Description": "Un atelier luxueux pour composer votre propre odeur et parfum personnalisé.", "Categorie": "Créatif & Culture", "Effort": "Faible", "Saison": "Toutes", "Tags": "odeur, luxe, intérieur, solo, manuel"},
    {"ID": 8, "NomPdt": "Bivouac hivernal en raquettes", "Description": "Aventure extrême dans la neige en montagne, incluant la marche en raquettes et la survie dans le froid.", "Categorie": "Aventure & Sport", "Effort": "Intense", "Saison": "Hiver", "Tags": "neige, froid, montagne, survie, sport"},
    {"ID": 9, "NomPdt": "Dégustation de vins dans une grotte", "Description": "Découvrez des vins d'exception dans une grotte souterraine lors d'une dégustation en groupe.", "Categorie": "Gastronomie", "Effort": "Faible", "Saison": "Toutes", "Tags": "vin, souterrain, insolite, groupe, dégustation"},
    {"ID": 10, "NomPdt": "Balade en chiens de traîneau", "Description": "Balade hivernale en famille tirée par des animaux passionnés dans la neige de la montagne.", "Categorie": "Nature & Découverte", "Effort": "Moyen", "Saison": "Hiver", "Tags": "animaux, neige, hiver, famille, nature"},
    {"ID": 11, "NomPdt": "Canoë-kayak au coucher du soleil", "Description": "Pagayez sur la rivière au coucher du soleil pour une expérience d'eau relaxante et sportive.", "Categorie": "Nature & Sport", "Effort": "Moyen", "Saison": "Été", "Tags": "eau, rivière, soleil, sport, détente"},
    {"ID": 12, "NomPdt": "Atelier cuisine locale et sauvage", "Description": "Apprenez à cuisiner avec des plantes sauvages trouvées dans la nature lors d'un atelier gastronomique.", "Categorie": "Gastronomie & Nature", "Effort": "Faible", "Saison": "Printemps/Été", "Tags": "cuisine, plantes, nature, groupe, manuel"},
    {"ID": 13, "NomPdt": "Nuit dans un igloo", "Description": "Dormez dans la neige à l'intérieur d'un igloo pour une nuit insolite dans le froid.", "Categorie": "Nature & Insolite", "Effort": "Moyen", "Saison": "Hiver", "Tags": "neige, froid, insolite, couple, montagne"},
    {"ID": 14, "NomPdt": "Initiation à la forge et coutellerie", "Description": "Travaillez le métal et le feu pour forger votre propre couteau artisanal.", "Categorie": "Créatif & Artisanat", "Effort": "Moyen", "Saison": "Toutes", "Tags": "métal, feu, manuel, artisanat, intérieur"},
    {"ID": 15, "NomPdt": "Saut à l'élastique depuis un viaduc", "Description": "Vivez l'adrénaline pure du vide en sautant depuis un pont ou un viaduc extrême.", "Categorie": "Aventure & Frissons", "Effort": "Moyen", "Saison": "Printemps/Été", "Tags": "vide, adrénaline, extrême, hauteur, solo"},
    {"ID": 16, "NomPdt": "Retraite yoga et méditation en monastère", "Description": "Trouvez le zen et le calme spirituel lors d'une retraite de méditation dans un monastère.", "Categorie": "Détente & Bien-être", "Effort": "Faible", "Saison": "Toutes", "Tags": "zen, calme, spirituel, intérieur, groupe"},
    {"ID": 17, "NomPdt": "Chasse au trésor urbaine (Escape Game)", "Description": "Résolvez des énigmes en groupe lors d'une chasse au trésor urbaine très fun en ville.", "Categorie": "Découverte & Jeu", "Effort": "Moyen", "Saison": "Toutes", "Tags": "ville, énigmes, groupe, fun, marche"},
    {"ID": 18, "NomPdt": "Atelier d'apiculture (découverte abeilles)", "Description": "Découvrez le monde des abeilles et la fabrication du miel en pleine nature avec votre famille.", "Categorie": "Nature & Découverte", "Effort": "Faible", "Saison": "Printemps/Été", "Tags": "animaux, miel, nature, famille, apprentissage"},
    {"ID": 19, "NomPdt": "Stage de photographie animalière", "Description": "Apprenez la patience et la photographie pour capturer des animaux sauvages en pleine nature.", "Categorie": "Nature & Créatif", "Effort": "Moyen", "Saison": "Toutes", "Tags": "photo, animaux, patience, nature, apprentissage"},
    {"ID": 20, "NomPdt": "Conduite d'une motoneige", "Description": "Conduisez un moteur puissant sur la neige à grande vitesse pour une aventure d'hiver pleine d'adrénaline.", "Categorie": "Aventure & Sport", "Effort": "Moyen", "Saison": "Hiver", "Tags": "vitesse, neige, moteur, adrénaline, montagne"}
]

df_pdt = pd.DataFrame(produits)
df_pdt.to_csv("produits.csv", index=False)

users = [{"IDUser": i, "NomUser": f"User_{i}"} for i in range(1, 21)]
df_users = pd.DataFrame(users)
df_users.to_csv("users.csv", index=False)

notes = []
end_date = datetime.now()
start_date = end_date - timedelta(days=730)
id_note = 1

for user in users:
    uid = user['IDUser']
    num_ratings = random.randint(5, 12)
    rated_pdts = random.sample(produits, num_ratings)
    
    for p in rated_pdts:
        rating = random.randint(1, 5)
        random_days = random.randint(0, 730)
        interaction_date = start_date + timedelta(days=random_days)
        
        notes.append({
            "IDNote": id_note,
            "IDUser": uid,
            "IDProduit": p["ID"],
            "Note": rating,
            "Timestamp": interaction_date.strftime("%Y-%m-%d %H:%M:%S")
        })
        id_note += 1

df_notes = pd.DataFrame(notes)
df_notes.to_csv("notes.csv", index=False)
print("Dataset regénéré avec les tags et les catégories !")

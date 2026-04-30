import psycopg2
import pandas as pd

def setup_database():

    print("Connexion PostgreSQL Render...")

    # 🔥 URL Render (remplace si besoin)
    DATABASE_URL = "postgresql://sr_j833_user:WXMnVS2PVorml3YjLDz9LhWRZ6VHgemr@dpg-d7pl468js32c73dva8k0-a.oregon-postgres.render.com:5432/sr_j833"

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        print("Connexion réussie ✔️")

        # 🔴 supprimer tables si existent
        cursor.execute("DROP TABLE IF EXISTS Notes CASCADE")
        cursor.execute("DROP TABLE IF EXISTS Users CASCADE")
        cursor.execute("DROP TABLE IF EXISTS Produit CASCADE")

        # 🟢 création tables

        cursor.execute("""
            CREATE TABLE Produit (
                ID INTEGER PRIMARY KEY,
                NomPdt VARCHAR(255),
                Description TEXT,
                Categorie VARCHAR(255),
                Effort VARCHAR(50),
                Saison VARCHAR(50),
                Tags TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE Users (
                IDUser INTEGER PRIMARY KEY,
                NomUser VARCHAR(255)
            )
        """)

        cursor.execute("""
            CREATE TABLE Notes (
                IDNote INTEGER PRIMARY KEY,
                IDUser INTEGER,
                IDProduit INTEGER,
                Note FLOAT,
                Timestamp TIMESTAMP,
                FOREIGN KEY (IDUser) REFERENCES Users(IDUser),
                FOREIGN KEY (IDProduit) REFERENCES Produit(ID)
            )
        """)

        print("Tables créées ✔️")

        # 📂 Charger CSV
        df_pdt = pd.read_csv("produits.csv")
        df_users = pd.read_csv("users.csv")
        df_notes = pd.read_csv("notes.csv")

        # 🟡 insert Produit
        for _, row in df_pdt.iterrows():
            cursor.execute("""
                INSERT INTO Produit VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, tuple(row))

        # 🟡 insert Users
        for _, row in df_users.iterrows():
            cursor.execute("""
                INSERT INTO Users VALUES (%s, %s)
            """, tuple(row))

        # 🟡 insert Notes
        for _, row in df_notes.iterrows():
            cursor.execute("""
                INSERT INTO Notes VALUES (%s, %s, %s, %s, %s)
            """, tuple(row))

        conn.commit()
        conn.close()

        print("Données insérées avec succès 🚀")

    except Exception as e:
        print("Erreur :", e)


if __name__ == "__main__":
    setup_database()
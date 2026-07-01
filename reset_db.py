import os
from sqlalchemy import create_engine
from src.utils.db import Base, DB_URL

def reset_db():
    engine = create_engine(DB_URL)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    print("Recreated schema perfectly")

if __name__ == "__main__":
    reset_db()

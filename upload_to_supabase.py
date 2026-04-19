import os
import psycopg2
from dotenv import load_dotenv

print("Loading .env file...")
load_dotenv()

db_url = os.getenv("DATABASE_URL")
if not db_url:
    print("ERROR: DATABASE_URL not found in .env")
    exit(1)

print(f"Connecting to: {db_url.split('@')[1]}")

try:
    # Connect to your new Supabase database!
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    cur = conn.cursor()

    print("Reading the 35MB echo_backup.sql file... This might take a few seconds.")
    with open("echo_backup.sql", "r", encoding="utf-8") as file:
        sql_commands = file.read()

    print("Executing SQL commands onto the Cloud... Please wait and DO NOT CLOSE this window.")
    cur.execute(sql_commands)
    
    print("\n✅ SUCCESS! All tables and data have been uploaded to Supabase!")
    
    cur.close()
    conn.close()

except Exception as e:
    print(f"\n❌ ERROR: {e}")

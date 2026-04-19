import psycopg2
import sys

pw = "J0Xy6hvddZ2Tkh7K"
host = "aws-0-eu-central-1.pooler.supabase.com"

tests = [
    # Supavisor transaction pool
    f"postgresql://postgres.ydnnqvupohfoibxlebnf:{pw}@{host}:6543/postgres",
    # Supavisor session pool
    f"postgresql://postgres.ydnnqvupohfoibxlebnf:{pw}@{host}:5432/postgres",
]

for url in tests:
    try:
        conn = psycopg2.connect(url)
        print(f"SUCCESS: {url}")
        conn.close()
        sys.exit(0)
    except Exception as e:
        print(f"FAILED for url {url}: {e}")

print("All tests failed.")

from google import genai
from google.genai import types

# 1. Initialize the client
client = genai.Client(api_key="AIzaSyDxdA2FJYsyDNjJo5G7jxOetDdgrwkzaF8")

# 2. Read your local text file
file_path = r"C:\Uni\4th Year\GP\Implementation\Video Generation\Pharaohs Docs Summarization and Description Task\docs\Rameses II.txt" 

with open(file_path, "r", encoding="utf-8") as f:
    file_content = f.read()

# 3. Configure the "strict" generation
config = types.GenerateContentConfig(
    system_instruction="You are a professional video scriptwriter. Output ONLY the narration script. Do not include scene descriptions, intros, or outros. Ensure the script length is exactly for a 1-minute video (approx. 150 words).",
    temperature=0.7
)

# 4. Generate the response
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[f"Based on the following text, write a 1-minute video narration script:\n\n{file_content}"],
    config=config
)

# 5. Output the result
print(response.text)
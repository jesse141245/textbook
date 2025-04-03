import argparse
import json
import os
import re
import fitz  # PyMuPDF
from dotenv import load_dotenv
from google import genai
from google.genai import types
from json_repair import repair_json

def extract_title(pdf_path: str) -> str:
    """
    Open the PDF and search the first 5 pages for the text span with the largest font size.
    That text is assumed to be the textbook title.
    """
    doc = fitz.open(pdf_path)
    max_size = 0
    title = ""
    pages_to_check = min(5, doc.page_count)
    for page_number in range(pages_to_check):
        page = doc.load_page(page_number)
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        size = span.get("size", 0)
                        text = span.get("text", "").strip()
                        if text and size > max_size:
                            max_size = size
                            title = text
    return title or "Unknown Textbook Title"

def clean_and_parse_json(json_line: str) -> dict:
    """
    Clean a JSON line by removing markdown formatting and control characters,
    then attempt to repair and parse it using json_repair.
    """
    cleaned = json_line.replace("```json", "").replace("```", "").strip()
    cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', cleaned)
    json_start = cleaned.find("{")
    json_end = cleaned.rfind("}") + 1
    if json_start == -1 or json_end == 0:
        raise ValueError("No valid JSON object found in line.")
    cleaned = cleaned[json_start:json_end]
    cleaned = re.sub(r'}\s*{', r'},{', cleaned)
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    try:
        repaired = repair_json(cleaned)
    except Exception as e:
        raise ValueError(f"json_repair failed: {e}")
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")

def main():
    load_dotenv()
    api_key = os.getenv("api_key")  

    parser = argparse.ArgumentParser(
        description="Extract textbook prerequisites and generate a multiple choice test."
    )
    parser.add_argument("pdf_file", help="Path to the input PDF.")
    parser.add_argument("-api-key", default=api_key, help="Gemini API key (or set api_key in .env).")
    parser.add_argument("--model-name", default="gemini-2.0-flash-thinking-exp-01-21", help="Gemini model name.")
    parser.add_argument("--output-file", default="prerequisites_test_mc_output.jsonl", help="Output file for test JSONL.")
    args = parser.parse_args()

    textbook_title = extract_title(args.pdf_file)
    print(f"Extracted textbook title: {textbook_title}")

    client = genai.Client(api_key=args.api_key)

    prompt = f"""
You are an expert educator. Given the textbook title "{textbook_title}", first determine the prerequisite topics that a student should know before studying this textbook.
Then, generate a test consisting of about 20 multiple choice questions that assess the student's understanding of these prerequisite topics.
For each question, provide:
  - "question": the text of the question,
  - "correct": the correct answer,
  - "wrong1": a plausible but incorrect answer,
  - "wrong2": another plausible but incorrect answer,
  - "wrong3": a third plausible but incorrect answer,
  - "subject": the prerequisite topic that the question tests.
Output each question as a JSON object (one per line) with no extra commentary or formatting.
"""

    response = client.models.generate_content(
        model=f"models/{args.model_name}",
        contents=[prompt]
    )

    if not response.text:
        print("No response received from the Gemini API.")
        return

    output_lines = response.text.strip().splitlines()
    valid_count = 0
    with open(args.output_file, "w", encoding="utf-8") as f:
        for line in output_lines:
            try:
                data = clean_and_parse_json(line)
                if all(k in data for k in ("question", "correct", "wrong1", "wrong2", "wrong3", "subject")):
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    valid_count += 1
                else:
                    print("Skipping line (missing keys):", line)
            except Exception as e:
                print("Skipping invalid JSON line:", line)
                print("Error:", e)

    print(f"Successfully saved {valid_count} JSON lines to {args.output_file}")

if __name__ == "__main__":
    main()

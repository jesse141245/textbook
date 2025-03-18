import argparse
import json
import os
import logging
import time
from dotenv import load_dotenv
from google import genai
from concurrent.futures import ThreadPoolExecutor, as_completed

class Config:
    def __init__(self, args):
        self.api_key = args.api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("Must provide a valid Gemini API key via --api-key or GEMINI_API_KEY env var.")
        self.model_name = args.model_name
        self.output_file = args.output_file
        self.chapters_dir = args.pdf_file.replace(".pdf", "")  # Still unused
        self.max_retries = args.max_retries
        # Correctly initialize the model object.  Rename 'client' to 'model' for clarity.
        self.model = genai.GenerativeModel(model_name=self.model_name, api_key=self.api_key)

def load_jsonl_file(file_path):
    """Generator to yield one JSON object per non-empty line."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl_file(records, output_path, config: Config):
    """Write a list of JSON records to a JSONL file."""
    with open(output_path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def rewrite_answer(question, answer, config: Config):
    """Rewrite the answer using the LLM model."""
    prompt = f"""You are an expert educator tasked with simplifying answers to the following questions.
Ensure there is reasoning and explanation in the answer, but if there is unnecessary information, remove it.
Look for phrases like "Alternatively," "But wait," or "Let me reconsider" that indicate redundant reasoning.
Identify and remove redundant reasoning steps.
Keep only the essential steps needed to arrive at the correct answer.
If there are multiple solutions, keep the first correct solution and one additional solution.
If the solution is already simple, do not simplify it further.

Question: {question}
Answer: {answer}

Rewritten Answer:"""

    attempt = 0
    while attempt < config.max_retries:
        attempt += 1
        try:
            # Call generate_content directly on the model object.
            response = config.model.generate_content(prompt)
            return response.text.strip()

        except Exception as e:
            logging.warning(f"Rewrite attempt {attempt}/{config.max_retries} failed: {e}")
            time.sleep(2 ** attempt)

    logging.error("Max retries reached for rewriting answer. Using original answer.")
    return answer

def run_rewrite(output_file, input_file, config: Config):
    """
    Reads from 'input_file', rewrites answers in 'qa' records in parallel,
    and writes the updated JSON to 'output_file'.
    """
    records = list(load_jsonl_file(input_file))
    updated_records = []

    def rewrite_answer_parallel(index, question, answer):
        logging.info(f"Rewriting answer for question #{index}: {question[:50]}...")
        return rewrite_answer(question, answer, config)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, record in enumerate(records):
            if record.get("metadata", {}).get("type") == "qa":
                question = record.get("instruction", "").strip()
                answer = record.get("output", "").strip()
                future = executor.submit(rewrite_answer_parallel, i, question, answer)
                futures.append((future, i))
            else:
                updated_records.append(record)

        for future, idx in futures:
            try:
                rewritten_answer = future.result()
                record = records[idx]
                record["output"] = rewritten_answer
                updated_records.append(record)
            except Exception as e:
                logging.error(f"Error processing record {idx}: {e}")
                updated_records.append(records[idx])

    updated_records.sort(key=lambda x: records.index(x) if x in records else -1) #Maintain record order
    write_jsonl_file(updated_records, output_file, config)
    logging.info(f"Rewritten answers saved to {output_file}")

def main():
    load_dotenv()
    api_key = os.getenv("api_key")
    parser = argparse.ArgumentParser(description="Rewrite answers in a JSONL file by removing unnecessary reasoning.")
    parser.add_argument("pdf_file", help="Path to the input PDF (not directly used).")
    parser.add_argument("-api-key", default= api_key,  help="Gemini API key (or set GEMINI_API_KEY env var).")
    parser.add_argument("--model-name", default="gemini-1.5-pro-002", help="Gemini model name.")
    parser.add_argument("--output-file", default="rewritten_output.jsonl", help="Path to the final JSONL.")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for the LLM call.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        config = Config(args)
    except ValueError as e:
        logging.error(e)
        return

    input_file = "input_data.jsonl"
    if not os.path.isfile(input_file):
        logging.error(f"Input JSONL file not found: {input_file}")
        return

    run_rewrite(config.output_file, input_file, config)

if __name__ == "__main__":
    main()
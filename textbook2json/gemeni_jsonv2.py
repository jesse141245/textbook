import re
import io
import time
import json
import base64
import argparse
import requests  
import os
import logging
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor
from json_repair import repair_json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PyPDF2 import PdfReader, PdfWriter
import copy
from rewrite import Config as RewriteConfig, run_rewrite
# --------------------------------------------------------------------
# Helper Function
# --------------------------------------------------------------------
def sanitize_filename(name: str) -> str:
    """
    Replaces characters that are invalid in Windows filenames with an underscore.
    Invalid characters: <>:"/\\|?*
    """
    return re.sub(r'[<>:"/\\|?*]', '_', name)

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
class Config:
    def __init__(self, args):
        self.api_key = args.api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError("Must provide a valid Gemini API key via --api-key or GEMINI_API_KEY env var.")

        self.model_name = args.model_name
        self.output_file = args.output_file
        self.max_retries = args.max_retries
        self.chapters_dir = args.pdf_file.replace(".pdf", "")

        self.prompt = """You are an expert educator and content extractor. Your task is to extract 
as many detailed and useful educational question-answer pairs and instructional notes as possible 
from the text chunk below. Please follow these steps carefully:

SUPER SUPER IMPORTANT FORMAT THE JSON OUTPUT CORRECTLY SO IT CAN BE SAVED.
0. IMPORTANT, PUT THE Q&A AND THE INSTRUCTIONAL NOTES IN LATEX FORM BE SURE NOT TO USE CONTROL CHARACTERS INVALID ESCAPE SEQUENCES SO I CAN PARSE THE JSON OUTPUT
1. Read the entire text thoroughly.
2. Identify every key concept, important detail, and nuance.
3. For each key point, generate a question that tests understanding and provide a comprehensive answer.
4. Aim for as many Q&A pairs as possible—e.g., 15 or more if the text allows.
5. Format your response strictly as valid JSON with exactly two keys:
   - "qa_pairs": an array of objects, each containing "question" and "answer" keys.
   - "instructional_notes": an array of strings.
6. Do not include any additional commentary or text outside of the JSON.
7. Your Q&A pairs should mimic the style of textbook problems with clear problem statements and detailed answers.
8. You should create your own Q&A's that are based on the topic.
9. Q&A pairs should also aim to implement some numerical/applied questions, questions that aren't proof based or theory based.
10. To answer questions, your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with ''} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|>
11. Your instructional notes should capture overarching themes like step-by-step methods.

JSON output:
"""

# --------------------------------------------------------------------
# PDF Splitting (using chapter overview pages)
# --------------------------------------------------------------------
def split_pdf_by_chapter(pdf_path: str, config) -> list:
    """
    Splits the PDF using chapter overview pages.
    Returns a list of tuples: ((chapter_title, subchapter_title), output_pdf_path)
    where subchapter_title is extracted from the overview; if not found, that chunk is skipped.
    """
    doc = fitz.open(pdf_path)
    os.makedirs(config.chapters_dir, exist_ok=True)

    chapter_overview_pages = []
    for page_index in range(doc.page_count):
        text = doc.load_page(page_index).get_text("text")
        if "CHAPTER OVERVIEW" in text:
            chapter_overview_pages.append(page_index)
    if not chapter_overview_pages:
        chapter_overview_pages = [0]
    chapter_overview_pages.append(doc.page_count)
    
    results = []  
    
    for i in range(len(chapter_overview_pages) - 1):
        chap_start = chapter_overview_pages[i]
        chap_end = chapter_overview_pages[i+1]
        
        overview_text = doc.load_page(chap_start).get_text("text")
        lines = [line.strip() for line in overview_text.splitlines() if line.strip()]
        
        try:
            overview_idx = next(idx for idx, ln in enumerate(lines) if "CHAPTER OVERVIEW" in ln.upper())
        except StopIteration:
            overview_idx = 0
        
        if overview_idx + 1 < len(lines):
            chapter_title_line = lines[overview_idx + 1]
        else:
            chapter_title_line = f"Chapter_{chap_start}"
        
        subchapter_lines = []
        for ln in lines[overview_idx + 2:]:
            if ln.startswith("This page titled") or "shared under" in ln:
                break
            subchapter_lines.append(ln)
        
        logging.info(f"Chapter overview (page {chap_start+1}): {chapter_title_line}")
        for sub in subchapter_lines:
            logging.info(f"  Found subchapter in overview: {sub}")
        
        subchapter_start_mapping = {}
        for sub_title in subchapter_lines:
            found_page = None
            for p in range(chap_start, chap_end):
                page_text = doc.load_page(p).get_text("text")
                if sub_title in page_text:
                    if p == chap_start:
                        continue
                    found_page = p
                    break
            subchapter_start_mapping[sub_title] = found_page
            if found_page is not None:
                logging.info(f"Subchapter '{sub_title}' starts on page {found_page+1}")
            else:
                logging.warning(f"Subchapter '{sub_title}' start not found.")
        
        subchapters_ordered = [(st, subchapter_start_mapping.get(st)) for st in subchapter_lines]
        
        for idx, (sub_title, start_page) in enumerate(subchapters_ordered):
            if start_page is None:
                continue
            if idx < len(subchapters_ordered) - 1:
                next_start = subchapters_ordered[idx+1][1]
                end_idx = next_start if next_start is not None else chap_end
            else:
                end_idx = chap_end
            end_page = end_idx  
            
            sub_doc = fitz.open()
            for p in range(start_page, end_page):
                sub_doc.insert_pdf(doc, from_page=p, to_page=p)
            
            sub_id = sub_title.split(":")[0].strip()
            filename = f"{sanitize_filename(sub_id)}.pdf"
            output_path = os.path.join(config.chapters_dir, filename)
            
            sub_doc.save(output_path)
            logging.info(f"Saved subchapter '{sub_id}' (pages {start_page+1}-{end_page}) to {filename}")
            results.append(((chapter_title_line, sub_id), output_path))
    
    return results

# --------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------

class PDFPipeline:
    def __init__(self, config: Config):
        self.config = config
        if os.path.exists(self.config.output_file):
            os.remove(self.config.output_file)
        # Correctly initialize the client:
        self.client = genai.Client(api_key=self.config.api_key)


    def _write_single_line(self, content: str, output_file: str):
        """Atomic write for individual JSON lines."""
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(content + "\n")


    def process_pdf(self, pdf_path: str):
            chapters = split_pdf_by_chapter(pdf_path, self.config)
            logging.info(f"Total {len(chapters)} chapter/subchapter PDF(s) created.")

            for i in range(0, len(chapters), 4):  # Process in batches of 4
                batch = chapters[i:i+4]
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {
                        executor.submit(self._call_llm, chapter_pdf_path, chapter_index): (chapter_index, chapter_pdf_path)
                        for chapter_index, chapter_pdf_path in batch
                    }
                    for future in futures:
                        chapter_index, chapter_pdf_path = futures[future]
                        try:
                            response_text = future.result()
                            self._write_jsonl_record(chapter_index, chapter_pdf_path, response_text)
                        except Exception as e:
                            logging.error(f"Error processing {chapter_index}: {e}")


    def _call_llm(self, chapter_pdf_path: str, chapter_index: tuple) -> str:
        with open(chapter_pdf_path, "rb") as f:
            chapter_bytes = f.read()

        attempt = 0
        while attempt < self.config.max_retries:
            attempt += 1
            try:
                chapter_part = types.Part.from_bytes(data=chapter_bytes, mime_type="application/pdf")
                response = self.client.models.generate_content(
                    model=self._fix_model_name(),
                    contents=[chapter_part, self.config.prompt]
                )
                if response.text is None:  # Check if .text is None
                    raise Exception("LLM response.text is None.")

                # Log token usage (important for cost tracking)
                tokens_used = (response.candidates[0].token_count if response.candidates else 0)
                logging.info(f"Chapter {chapter_index}: {tokens_used} tokens used")
                return response.text

            except Exception as e:
                logging.warning(f"LLM call failed for {chapter_index}, attempt {attempt}/{self.config.max_retries}: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff

        logging.error(f"Giving up on {chapter_index} after {self.config.max_retries} retries.")
        return ""  # Return empty string on failure


    def parse_json(self, response_text: str) -> dict:
        """
        Parses the LLM response text, attempting to extract a valid JSON object.
        Handles various common issues, including extra text, invalid characters,
        and incorrect delimiters.  Returns a dictionary on success, or raises
        a *detailed* exception on failure.
        """
        if response_text is None:
            raise ValueError("Response text is None.")

        # 1. Remove common LLM prefixes/suffixes (like ```json ... ```)
        cleaned = response_text.replace('```json', '').replace('```', '').strip()

        # 2. Remove invalid control characters.
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', cleaned)

        # 3. Try to find the *first* '{' and the *last* '}' to handle extra text.
        json_start = cleaned.find('{')
        json_end = cleaned.rfind('}') + 1  # +1 to *include* the last '}'

        if json_start == -1 or json_end == 0:
            raise ValueError(f"No valid JSON object found in response.  First 200 chars: {cleaned[:200]}")
        cleaned = cleaned[json_start:json_end]

        # 4. Try to fix common JSON errors (like missing commas between objects).
        cleaned = re.sub(r'}\s*{', r'},{', cleaned)  # Add commas between adjacent objects
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned) # Remove trailing commas

        # 5. Use json_repair to handle more complex errors.
        try:
            repaired = repair_json(cleaned)
        except Exception as e:
            raise ValueError(f"json_repair failed: {e}.  Original (cleaned) text (first 200 chars): {cleaned[:200]}")

        # 6. Attempt to parse as JSON.
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON after repair: {e}\nRepaired JSON (first 300 chars): {repaired[:300]}\nOriginal (cleaned) text (first 300 chars): {cleaned[:300]}")

    def _write_jsonl_record(self, chunk_index: tuple, chunk_pdf_path: str, response_text: str):
        chapter_title, subchapter_title = chunk_index
        safe_chapter = sanitize_filename(chapter_title.replace(" ", "_"))

        chapter_folder = os.path.join(self.config.chapters_dir, 'jsonl')
        os.makedirs(chapter_folder, exist_ok=True)
        
        if subchapter_title:
            safe_sub = sanitize_filename(subchapter_title.replace(" ", "_"))
            filename = f"{safe_chapter}_{safe_sub}.jsonl"
        else:
            filename = f"{safe_chapter}.jsonl"
        
        output_file = os.path.join(chapter_folder, filename)
        
        try:
            response_data = self.parse_json(response_text) #parse the json

            # Handle a list of JSON objects
            if isinstance(response_data, list):
                combined = {"qa_pairs": [], "instructional_notes": []}
                for item in response_data: #combine each dict
                    if isinstance(item, dict):
                        combined["qa_pairs"].extend(item.get("qa_pairs", []))
                        combined["instructional_notes"].extend(item.get("instructional_notes", []))
                response_data = combined  # Use the combined dictionary

            # Check if response is dict with expected structure
            if not isinstance(response_data, dict) or not all(k in response_data for k in ["qa_pairs", "instructional_notes"]):
                raise ValueError(f"Parsed JSON is not a dictionary or missing keys: {response_data.keys() if isinstance(response_data, dict) else 'Not a dict'}")
            
            # Write QA pairs.
            for qa in response_data.get("qa_pairs", []):
                if not isinstance(qa, dict) or "question" not in qa or "answer" not in qa:
                    logging.warning(f"Skipping invalid QA pair: {qa}")
                    continue

                entry = {
                    "instruction": qa.get("question", "").strip(),
                    "input": "",
                    "output": qa.get("answer", "").strip(),
                    "metadata": {
                        "chapter": chapter_title,
                        "subchapter": subchapter_title,
                        "source": chunk_pdf_path,
                        "type": "qa"
                    }
                }
                self._write_single_line(json.dumps(entry, ensure_ascii=False), output_file)
            
            # Write instructional notes.
            for note_idx, note in enumerate(response_data.get("instructional_notes", [])):
                if not isinstance(note, str):
                    logging.warning(f"Skipping invalid instructional note (not a string): {note}")
                    continue
                entry = {
                    "instruction": note.strip(),
                    "input": "",
                    "output": "",  # No output for notes
                    "metadata": {
                        "chapter": chapter_title,
                        "subchapter": subchapter_title,
                        "source": chunk_pdf_path,
                        "type": "note",
                        "note_index": note_idx + 1
                    }
                }
                self._write_single_line(json.dumps(entry, ensure_ascii=False), output_file)
        except Exception as e:
            # Log the error with as much context as possible *including the raw response*
            logging.error(f"Error processing chunk {chunk_index} from {chunk_pdf_path}: {e}")
            error_entry = {
                "instruction": "UNHANDLED PROCESSING ERROR",
                "input": "",
                "output": str(e),  # Include the full exception message
                "raw_response": response_text[:500] if response_text else "No response",  # Include part of the raw response
                "metadata": {
                    "source": os.path.basename(chunk_pdf_path),
                    "type": "error",
                    "chunk_index": chunk_index
                }
            }
            self._write_single_line(json.dumps(error_entry, ensure_ascii=False), output_file)

        # --- Rewriting (Optional) ---
        chapter_folder_re = os.path.join(self.config.chapters_dir, 'rewritten_jsonl')
        os.makedirs(chapter_folder_re, exist_ok=True)
        rewriting_output = os.path.join(chapter_folder_re, f"{safe_chapter}_rewritten.jsonl")

        new_config = copy.copy(self.config)
        run_rewrite(rewriting_output, output_file, new_config)

    def _fix_model_name(self) -> str:
        if self.config.model_name.startswith("models/"):
            return self.config.model_name
        else:
            return f"models/{self.config.model_name}"
# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    load_dotenv()
    api_key = os.getenv("api_key")  # Correctly load from .env
    parser = argparse.ArgumentParser(description="Split PDF, extract Q&A, and save as JSONL.")
    parser.add_argument("pdf_file", help="Path to the input PDF.")
    parser.add_argument("-api-key", default=api_key, help="Gemini API key (or set GEMINI_API_KEY env var).")
    parser.add_argument("--model-name", default="gemini-2.0-flash-thinking-exp-01-21", help="Gemini model name.")
    parser.add_argument("--output-file", default="pdf_chunks_output.jsonl", help="(Not used for direct output)")
    parser.add_argument("--max-retries", type=int, default=3, help="Number of retries for LLM calls.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        config = Config(args)
    except ValueError as e:
        logging.error(e)
        return

    if not os.path.isfile(args.pdf_file):
        logging.error(f"PDF file not found: {args.pdf_file}")
        return

    pipeline = PDFPipeline(config)
    pipeline.process_pdf(args.pdf_file)

if __name__ == "__main__":
    main()
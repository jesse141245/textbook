#!/usr/bin/env python
import re
import io
import time
import json
import base64
import argparse
import requests
import os
import logging
import fitz # PyMuPDF
import copy
import threading
from collections import defaultdict # Added for grouping
from concurrent.futures import ThreadPoolExecutor, as_completed # Added as_completed
from json_repair import repair_json
from dotenv import load_dotenv
from google import genai
from google.genai import types

from rewrite import Config as RewriteConfig, run_rewrite
from train_model_on_chapter import train_model_on_chapter


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Helper Function
# --------------------------------------------------------------------
def sanitize_filename(name: str) -> str:
    """Replaces invalid filename characters with underscores."""
    # Replace spaces first for better readability
    name = name.replace(" ", "_")
    # Remove or replace other invalid characters
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Remove leading/trailing underscores/spaces that might result
    name = name.strip('_ ')
    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name)
    # Limit length if necessary (optional)
    # max_len = 100
    # name = name[:max_len]
    return name

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
class Config:
    def __init__(self, args):
        load_dotenv() # Ensure .env is loaded
        self.api_key = args.api_key or os.getenv("gemini_api_key") # Simpler getenv
        if not self.api_key:
            raise ValueError("Must provide a valid Gemini API key via --api-key or GEMINI_API_KEY env var.")
        self.model_name = args.model_name
        self.original_pdf_filename = os.path.basename(args.pdf_file)
        self.textbook_name = sanitize_filename(os.path.splitext(self.original_pdf_filename)[0])
        self.work_dir = self.textbook_name + "_data"
        self.split_pdf_dir = os.path.join(self.work_dir, "split_pdfs")
        self.jsonl_output_dir = os.path.join(self.work_dir, "jsonl_output")
        self.rewritten_jsonl_dir = os.path.join(self.work_dir, "rewritten_jsonl")
        self.status_dir = os.path.join(self.work_dir, "status")
        self.trained_models_dir = os.path.join(self.work_dir, "trained_models")

        self.max_retries = args.max_retries
        self.prompt = """You are an expert educator and content extractor. Your task is to extract
as many detailed and useful educational question-answer pairs and instructional notes as possible
from the text chunk below. Please follow these steps carefully:

SUPER SUPER IMPORTANT FORMAT THE JSON OUTPUT CORRECTLY SO IT CAN BE SAVED.
0. IMPORTANT, PUT THE Q&A AND THE INSTRUCTIONAL NOTES IN LATEX FORM BE SURE NOT TO USE CONTROL CHARACTERS INVALID ESCAPE SEQUENCES SO I CAN PARSE THE JSON OUTPUT
1. Read the entire text thoroughly.
2. Identify every key concept, important detail, and nuance.
3. For each key point, generate a question that tests understanding and provide a comprehensive answer.
4. Aim for as many Q&A pairs as possible—e.g., 20 or more if the text allows.
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
# PDF Splitting
# --------------------------------------------------------------------
def split_pdf_by_chapter(pdf_path: str, config: Config) -> dict[str, list[tuple[str | None, str]]]:
    """
    Splits the PDF by top-level TOC entries (chapters).
    Returns a dictionary mapping chapter titles to a list of tuples:
    [(subchapter_title | None, subchapter_pdf_path), ...].
    Chapters without subchapters will have one entry with subchapter_title=None.
    """
    chapters_data = defaultdict(list)
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"Failed to open PDF {pdf_path}: {e}")
        return {}

    os.makedirs(config.split_pdf_dir, exist_ok=True)

    toc = doc.get_toc()
    if not toc:
        logging.warning(f"No Table of Contents found in PDF: {pdf_path}. Cannot split by chapter.")
        return {}

    processed_chapters_info = []
    current_chapter_info = None
    skip_keywords = ["answer key", "glossary", "index", "preface", "blank_page", "blank", "appendix", "acknowledgment", 'contents', 'table of contents']

    for i, entry in enumerate(toc):
        level, title, page = entry
        page_idx = page - 1 

        if level == 1: 
            if any(kw in title.lower() for kw in skip_keywords):
                logging.info(f"Skipping potential chapter '{title}' (page {page}) due to skip keywords.")
                current_chapter_info = None # Ensure subsequent subchapters aren't added
                continue

            if current_chapter_info:
                current_chapter_info["end_page"] = page_idx

            current_chapter_info = {"title": title, "start_page": page_idx, "end_page": doc.page_count, "subchapters": []}
            processed_chapters_info.append(current_chapter_info)

        elif level > 1 and current_chapter_info:
            current_chapter_info["subchapters"].append({"title": title, "page": page_idx})

    for i, chapter_info in enumerate(processed_chapters_info):
        chapter_title = chapter_info["title"]
        safe_chapter_title = sanitize_filename(chapter_title)
        chapter_start = chapter_info["start_page"]
        chapter_end = chapter_info["end_page"] 

        if chapter_end <= chapter_start:
            logging.warning(f"Chapter '{chapter_title}' has no pages (start: {chapter_start+1}, end: {chapter_end}); skipping.")
            continue

        if chapter_info["subchapters"]:
            subs = sorted(chapter_info["subchapters"], key=lambda x: x["page"]) # Ensure sorted
            last_sub_end = chapter_start
            for j, sub_info in enumerate(subs):
                sub_title = sub_info["title"]
                safe_sub_title = sanitize_filename(sub_title)
                sub_start = sub_info["page"]

                if j + 1 < len(subs):
                    sub_end = subs[j+1]["page"]
                else:
                    sub_end = chapter_end

                if sub_end <= sub_start:
                    logging.warning(f"Subchapter '{sub_title}' of chapter '{chapter_title}' has invalid page range (start: {sub_start+1}, end: {sub_end}); skipping.")
                    continue

                sub_doc = fitz.open()
                try:
                    sub_doc.insert_pdf(doc, from_page=sub_start, to_page=sub_end - 1) 
                    filename = f"{safe_chapter_title}_{safe_sub_title}.pdf"
                    output_path = os.path.join(config.split_pdf_dir, filename)
                    sub_doc.save(output_path)
                    sub_doc.close()
                    logging.info(f"  Saved subchapter chunk '{sub_title}' (pages {sub_start+1}-{sub_end}) to {filename}")
                    chapters_data[chapter_title].append((sub_title, output_path))
                    last_sub_end = sub_end
                except Exception as e:
                     logging.error(f"  Error processing subchapter '{sub_title}' pages {sub_start+1}-{sub_end}: {e}")


        else: 
            chapter_doc = fitz.open()
            try:
                chapter_doc.insert_pdf(doc, from_page=chapter_start, to_page=chapter_end - 1)
                filename = f"{safe_chapter_title}_CHAPTER.pdf" 
                output_path = os.path.join(config.split_pdf_dir, filename)
                chapter_doc.save(output_path)
                chapter_doc.close()
                logging.info(f"Saved chapter '{chapter_title}' (pages {chapter_start+1}-{chapter_end}) to {filename}")
                chapters_data[chapter_title].append((None, output_path)) 
            except Exception as e:
                logging.error(f"Error processing chapter '{chapter_title}' pages {chapter_start+1}-{chapter_end}: {e}")

    doc.close()
    return chapters_data

# --------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------
class PDFPipeline:
    def __init__(self, config: Config):
        self.config = config
        os.makedirs(self.config.work_dir, exist_ok=True)
        os.makedirs(self.config.jsonl_output_dir, exist_ok=True)
        os.makedirs(self.config.rewritten_jsonl_dir, exist_ok=True)
        os.makedirs(self.config.status_dir, exist_ok=True)

        try:
            if config.api_key:
                 self.client = genai.Client(api_key=self.config.api_key)
            else:
                 self.client = None
                 logging.warning("No API key provided, LLM calls will be skipped.")
        except Exception as e:
            logging.error(f"Failed to initialize Google GenAI Client: {e}")
            self.client = None 

    def _atomic_write_line(self, content: str, output_file: str):
        """Appends a line to a file atomically (generally safe for line-based appends)."""
        try:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(content + "\n")
        except Exception as e:
            logging.error(f"Failed to write line to {output_file}: {e}")


    def _update_progress(self, current_step: int, total_steps: int):
        """Updates the progress.json file."""
        progress_file = os.path.join(self.config.status_dir, "progress.json")
        percentage = (current_step / total_steps) * 100 if total_steps > 0 else 0
        progress_data = {
            # Assuming epoch remains 0 for this single run process
            "epoch": 0,
            "step": current_step,
            "total_steps": total_steps,
            "percentage": round(percentage, 2)
        }
        try:
            with open(progress_file, "w") as f:
                json.dump(progress_data, f)
            logging.debug(f"Progress updated: Step {current_step}/{total_steps} ({percentage:.2f}%)")
        except Exception as e:
            logging.error(f"Failed to update progress file {progress_file}: {e}")

    def process_pdf(self, pdf_path: str):
        """Processes the PDF chapter by chapter."""
        logging.info(f"Starting processing for: {self.config.original_pdf_filename}")
        logging.info(f"Textbook Name (Sanitized): {self.config.textbook_name}")
        logging.info(f"Working Directory: {self.config.work_dir}")

        chapters_to_process = split_pdf_by_chapter(pdf_path, self.config)
        num_chapters = len(chapters_to_process)
        logging.info(f"Found {num_chapters} chapters to process.")

        if num_chapters == 0:
            logging.warning("No chapters found or PDF could not be processed. Exiting.")
            self._update_progress(0, 0) # Indicate nothing to process
            return

        # Initialize progress
        self._update_progress(0, num_chapters)
        chapters_processed_count = 0

        # Process each chapter
        for chapter_title, chunks in chapters_to_process.items():
            safe_chapter_title = sanitize_filename(chapter_title)
            logging.info(f"\n--- Processing Chapter: '{chapter_title}' ({len(chunks)} chunk(s)) ---")

            # Define the single JSONL output file for this entire chapter
            chapter_jsonl_file = os.path.join(self.config.jsonl_output_dir, f"{self.config.textbook_name}_{safe_chapter_title}.jsonl")
            # Clear the file if it exists from a previous run
            if os.path.exists(chapter_jsonl_file):
                logging.warning(f"Output file {chapter_jsonl_file} exists, clearing before processing.")
                os.remove(chapter_jsonl_file)

            # Process all chunks (subchapters) of this chapter in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                # {future: (subchapter_title, chunk_pdf_path)}
                future_to_chunk = {
                    executor.submit(self._call_llm, chunk_pdf_path, (chapter_title, sub_title)): (sub_title, chunk_pdf_path)
                    for sub_title, chunk_pdf_path in chunks
                }

                for future in as_completed(future_to_chunk):
                    sub_title, chunk_pdf_path = future_to_chunk[future]
                    chunk_index = (chapter_title, sub_title) # For logging/metadata
                    try:
                        response_text = future.result() # Get LLM response text
                        if response_text: # Only process if we got a response
                             # Write the result to the *chapter's* aggregated JSONL file
                            self._write_chunk_to_chapter_jsonl(
                                chunk_index,
                                chunk_pdf_path,
                                response_text,
                                chapter_jsonl_file # Pass the target file
                            )
                        else:
                            logging.warning(f"No response text received for chunk {chunk_index}, skipping write.")
                    except Exception as e:
                        logging.error(f"Error processing result for chunk {chunk_index} from {chunk_pdf_path}: {e}")
                        # Log error to the chapter file as well?
                        error_entry = {
                             "instruction": f"ERROR processing chunk: Chapter='{chapter_title}', Subchapter='{sub_title}'",
                             "input": chunk_pdf_path,
                             "output": f"Error: {e}",
                             "metadata": {"type": "processing_error"}
                        }
                        self._atomic_write_line(json.dumps(error_entry, ensure_ascii=False), chapter_jsonl_file)


            logging.info(f"--- Finished processing chunks for Chapter: '{chapter_title}' ---")

            # --- Post-Chapter Processing (Rewriting & Training) ---
            if os.path.exists(chapter_jsonl_file): # Only proceed if the file was created
                # 1. Rewriting
                # rewritten_chapter_jsonl_file = os.path.join(
                #     self.config.rewritten_jsonl_dir,
                #     f"{self.config.textbook_name}_{safe_chapter_title}_rewritten.jsonl"
                # )
                # logging.info(f"Starting rewrite for chapter '{chapter_title}'...")
                # try:
                #     # Assuming run_rewrite takes output path, input path, config
                #     # Create a config suitable for rewrite if needed, or reuse/copy
                #     rewrite_config = copy.copy(self.config) # Or specific RewriteConfig
                #     run_rewrite(rewritten_chapter_jsonl_file, chapter_jsonl_file, rewrite_config)
                #     logging.info(f"Rewrite complete for chapter '{chapter_title}'. Output: {rewritten_chapter_jsonl_file}")
                # except Exception as e:
                #     logging.error(f"Rewrite failed for chapter '{chapter_title}': {e}")
                
                # 2. Training (using the original chapter JSONL)
                logging.info(f"Starting training for chapter '{chapter_title}' using {chapter_jsonl_file}...")
                try:
                    # Assuming train_model_on_chapter takes textbook name and jsonl path
                    train_model_on_chapter(self.config.trained_models_dir, chapter_jsonl_file)
                    logging.info(f"Training complete for chapter '{chapter_title}'.")
                except Exception as e:
                    logging.error(f"Training failed for chapter '{chapter_title}': {e}")
            else:
                 logging.warning(f"Skipping rewrite and training for chapter '{chapter_title}' as no JSONL data was generated ({chapter_jsonl_file}).")

            # Update progress after completing a chapter
            chapters_processed_count += 1
            self._update_progress(chapters_processed_count, num_chapters)

        logging.info(f"\n=== Processing finished for {self.config.original_pdf_filename} ===")


    def _call_llm(self, chunk_pdf_path: str, chunk_index: tuple) -> str:
        """Calls the LLM API for a single PDF chunk."""
        chapter_title, sub_title = chunk_index
        log_prefix = f"Chapter '{chapter_title}'"
        if sub_title:
            log_prefix += f" / Subchapter '{sub_title}'"
        log_prefix += f" ({os.path.basename(chunk_pdf_path)})"

        if not self.client:
            logging.warning(f"{log_prefix}: Skipping LLM call - client not initialized.")
            return ""

        try:
            with open(chunk_pdf_path, "rb") as f:
                chunk_bytes = f.read()
        except Exception as e:
            logging.error(f"{log_prefix}: Failed to read PDF chunk file {chunk_pdf_path}: {e}")
            return ""

        attempt = 0
        while attempt < self.config.max_retries:
            attempt += 1
            try:
                logging.info(f"{log_prefix}: Calling LLM (Attempt {attempt}/{self.config.max_retries})...")
                # Ensure mime_type is correct for your model/API
                chunk_part = types.Part.from_bytes(data=chunk_bytes, mime_type="application/pdf")
                response = self.client.models.generate_content( # Use generate_content directly
                    model=self._fix_model_name(),
                    contents=[chunk_part, self.config.prompt],
                    # Add generation_config if needed (e.g., temperature, max_output_tokens)
                    # generation_config=genai.types.GenerationConfig(...)
                )

                # Simple check if response has text (adapt based on actual API response structure)
                if hasattr(response, 'text') and response.text:
                     # Basic token count check if available
                    tokens_used = 0
                    if hasattr(response, 'usage_metadata') and response.usage_metadata:
                        tokens_used = response.usage_metadata.total_token_count # Example attribute
                    elif hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0], 'token_count'):
                         tokens_used = response.candidates[0].token_count # Old example attribute

                    logging.info(f"{log_prefix}: LLM call successful. Tokens used: {tokens_used if tokens_used else 'N/A'}")
                    return response.text
                elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                     # Handle cases where content was blocked
                     logging.error(f"{log_prefix}: LLM call blocked. Reason: {response.prompt_feedback.block_reason}")
                     return "" # Return empty on block
                else:
                    # Handle cases where response exists but has no text content
                    logging.warning(f"{log_prefix}: LLM response received but contains no text.")
                    # Log the response structure might be helpful here for debugging
                    # logging.debug(f"Non-text response structure: {response}")
                    return ""

            except Exception as e:
                # Catch API errors, connection issues etc.
                logging.warning(f"{log_prefix}: LLM call failed (Attempt {attempt}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries:
                    wait_time = 2 ** attempt
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                # Consider logging specific error types differently if needed

        logging.error(f"{log_prefix}: Giving up LLM call after {self.config.max_retries} retries.")
        return "" # Return empty string after all retries fail


    def _parse_json(self, response_text: str, context: str = "") -> dict | None:
        """
        Parses the LLM response text into a dictionary. Uses json_repair.
        Returns None on failure.
        """
        if not response_text:
            logging.warning(f"{context}: Cannot parse empty response text.")
            return None

        try:
            # Basic cleaning: remove markdown code fences
            cleaned = response_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            # Attempt repair
            repaired = repair_json(cleaned)
            # Attempt final parse
            parsed_data = json.loads(repaired)
            if not isinstance(parsed_data, dict):
                 logging.warning(f"{context}: Repaired JSON is not a dictionary ({type(parsed_data)}).")
                 return None
            # Basic validation for expected keys
            if not all(k in parsed_data for k in ["qa_pairs", "instructional_notes"]):
                logging.warning(f"{context}: Parsed JSON missing expected keys 'qa_pairs' or 'instructional_notes'. Found: {list(parsed_data.keys())}")
                # Optionally try to salvage if keys partially exist? For now, require both.
                return None
            return parsed_data
        except Exception as e:
            logging.error(f"{context}: Failed to parse JSON response: {e}. Response (first 300 chars): {response_text[:300]}")
            return None

    def _write_chunk_to_chapter_jsonl(self, chunk_index: tuple, chunk_pdf_path: str, response_text: str, chapter_jsonl_file: str):
        """Parses LLM response for a chunk and appends valid entries to the chapter's JSONL file."""
        chapter_title, subchapter_title = chunk_index
        context = f"Chapter '{chapter_title}' / Subchapter '{subchapter_title or 'N/A'}' ({os.path.basename(chunk_pdf_path)})"

        parsed_data = self._parse_json(response_text, context)

        if not parsed_data:
            logging.warning(f"{context}: Skipping write due to parsing failure.")
            # Write error info to the file?
            error_entry = {
                "instruction": f"PARSING ERROR: Chapter='{chapter_title}', Subchapter='{subchapter_title or 'N/A'}'",
                "input": os.path.basename(chunk_pdf_path),
                "output": f"Failed to parse LLM response.",
                "raw_response_preview": response_text[:500] if response_text else "No response",
                "metadata": {"type": "parsing_error"}
            }
            self._atomic_write_line(json.dumps(error_entry, ensure_ascii=False), chapter_jsonl_file)
            return

        items_written = 0
        # Process QA pairs
        qa_pairs = parsed_data.get("qa_pairs", [])
        if not isinstance(qa_pairs, list):
             logging.warning(f"{context}: 'qa_pairs' field is not a list, skipping QA pairs.")
             qa_pairs = []

        for qa in qa_pairs:
            if not isinstance(qa, dict) or "question" not in qa or "answer" not in qa:
                logging.warning(f"{context}: Skipping invalid QA pair format: {type(qa)}")
                continue

            entry = {
                # Combine chapter/subchapter/question for clarity
                "instruction": f"Chapter: {chapter_title}\nSubchapter: {subchapter_title or 'Chapter Main'}\nQuestion: {qa.get('question', '').strip()}",
                "input": "", # Input often not needed for this format
                "output": qa.get("answer", "").strip(),
                "metadata": {
                    "textbook": self.config.textbook_name,
                    "chapter": chapter_title,
                    "subchapter": subchapter_title, # Keep None if no subchapter
                    "source_pdf_chunk": os.path.basename(chunk_pdf_path),
                    "type": "qa"
                }
            }
            self._atomic_write_line(json.dumps(entry, ensure_ascii=False), chapter_jsonl_file)
            items_written += 1

        # Process instructional notes
        notes = parsed_data.get("instructional_notes", [])
        if not isinstance(notes, list):
            logging.warning(f"{context}: 'instructional_notes' field is not a list, skipping notes.")
            notes = []

        for note_idx, note in enumerate(notes):
            if not isinstance(note, str):
                logging.warning(f"{context}: Skipping invalid instructional note (not a string): {type(note)}")
                continue
            entry = {
                 # Combine chapter/subchapter/note for clarity
                "instruction": f"Chapter: {chapter_title}\nSubchapter: {subchapter_title or 'Chapter Main'}\nNote: {note.strip()}",
                "input": "",
                "output": "", # Notes usually don't have a direct 'output' in this format
                "metadata": {
                    "textbook": self.config.textbook_name,
                    "chapter": chapter_title,
                    "subchapter": subchapter_title,
                    "source_pdf_chunk": os.path.basename(chunk_pdf_path),
                    "type": "note",
                    "note_index": note_idx + 1
                }
            }
            self._atomic_write_line(json.dumps(entry, ensure_ascii=False), chapter_jsonl_file)
            items_written += 1

        logging.info(f"{context}: Appended {items_written} items to {os.path.basename(chapter_jsonl_file)}")


    def _fix_model_name(self) -> str:
        """Ensures model name has the 'models/' prefix if needed."""
        # Handle case where model_name might be None or empty
        if not self.config.model_name:
             # Default model or raise error? Let's default for now.
             logging.warning("Model name is empty in config, defaulting to 'models/gemini-1.5-flash'")
             return "models/gemini-1.5-flash" # Or appropriate default
        if self.config.model_name.startswith("models/"):
            return self.config.model_name
        else:
            # Simple heuristic, might need adjustment based on actual model naming patterns
            return f"models/{self.config.model_name}"

# --------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------
def main():
    load_dotenv() # Load environment variables from .env file

    parser = argparse.ArgumentParser(description="Split PDF by chapter, extract Q&A/Notes using LLM, rewrite, and train per chapter.")
    parser.add_argument("pdf_file", help="Path to the input PDF file.")
    parser.add_argument("--api-key", default=os.getenv("gemini_api_key"), help="Gemini API key (overrides GEMINI_API_KEY env var).")
    parser.add_argument("--model-name", default="gemini-1.5-flash", help="Gemini model name (e.g., gemini-1.5-pro, gemini-1.5-flash).")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for failed LLM API calls.")

    args = parser.parse_args()

    # Setup config (this will load API key from args or .env)
    try:
        config = Config(args)
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
        return # Exit if config fails (e.g., no API key)

    if not os.path.isfile(args.pdf_file):
        logging.error(f"Input PDF file not found: {args.pdf_file}")
        return

    # --- Start Pipeline ---
    pipeline = PDFPipeline(config)
    pipeline.process_pdf(args.pdf_file)

if __name__ == "__main__":
    main()
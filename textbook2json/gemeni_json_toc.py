import re
import io
import time
import json
import base64
import argparse
import requests
import os
import logging
import fitz  
from concurrent.futures import ThreadPoolExecutor
from json_repair import repair_json
from dotenv import load_dotenv
from google import genai
from google.genai import types

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
        self.chapters_dir = args.pdf_file.replace(".pdf", "")
        self.max_retries = args.max_retries  
        
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
# PDF Splitting using the TOC only
# --------------------------------------------------------------------
def split_pdf_by_chapter(pdf_path: str, config) -> list:
    """
    Splits the PDF using the TOC entries.
    Returns a list of tuples: ((chapter_title, subchapter_title), output_pdf_path)
    where subchapter_title is None if not applicable.
    """
    doc = fitz.open(pdf_path)
    os.makedirs(config.chapters_dir, exist_ok=True)
    
    toc = doc.get_toc()
    if not toc:
        logging.warning("No TOC found in PDF.")
        return []
    
    skip_keywords = ["answer key", "glossary", "index", "preface"]
    
    processed_chapters = []
    current_chapter = None
    for entry in toc:
        level, title, page = entry
        page_idx = page - 1 
        
        if level == 1:
            if any(kw in title.lower() for kw in skip_keywords):
                logging.info(f"Skipping chapter '{title}' (page {page}) due to skip keywords.")
                current_chapter = None
                continue
            current_chapter = {"title": title, "page": page_idx, "subchapters": []}
            processed_chapters.append(current_chapter)
        elif level >= 2:
            if current_chapter is None:
                continue
            current_chapter["subchapters"].append({"title": title, "page": page_idx})
    
    results = []
    for i, chapter in enumerate(processed_chapters):
        chapter_title = chapter["title"]
        chapter_start = chapter["page"]
        if i + 1 < len(processed_chapters):
            chapter_end = processed_chapters[i + 1]["page"]
        else:
            chapter_end = doc.page_count

        if chapter_end <= chapter_start:
            logging.warning(f"Chapter '{chapter_title}' has no pages (start: {chapter_start+1}, end: {chapter_end+1}); skipping.")
            continue

        if chapter["subchapters"]:
            subs = chapter["subchapters"]
            for j, sub in enumerate(subs):
                sub_title = sub["title"]
                sub_start = sub["page"]
                if j + 1 < len(subs):
                    sub_end = subs[j + 1]["page"]
                else:
                    sub_end = chapter_end

                if sub_end <= sub_start:
                    logging.warning(f"Subchapter '{sub_title}' of chapter '{chapter_title}' has no pages (start: {sub_start+1}, end: {sub_end+1}); skipping.")
                    continue

                sub_doc = fitz.open()
                for p in range(sub_start, sub_end):
                    sub_doc.insert_pdf(doc, from_page=p, to_page=p)
                
                safe_chapter = sanitize_filename(chapter_title.replace(" ", "_"))
                safe_sub = sanitize_filename(sub_title.replace(" ", "_"))
                filename = f"{safe_chapter}_{safe_sub}.pdf"
                output_path = os.path.join(config.chapters_dir, filename)
                
                sub_doc.save(output_path)
                logging.info(f"Saved subchapter '{sub_title}' of chapter '{chapter_title}' (pages {sub_start+1}-{sub_end}) to {filename}")
                results.append(((chapter_title, sub_title), output_path))
        else:
            sub_doc = fitz.open()
            for p in range(chapter_start, chapter_end):
                sub_doc.insert_pdf(doc, from_page=p, to_page=p)
            safe_chapter = sanitize_filename(chapter_title.replace(" ", "_"))
            filename = f"{safe_chapter}.pdf"
            output_path = os.path.join(config.chapters_dir, filename)
            
            sub_doc.save(output_path)
            logging.info(f"Saved chapter '{chapter_title}' (pages {chapter_start+1}-{chapter_end}) to {filename}")
            results.append(((chapter_title, None), output_path))
    
    return results

# --------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------
class PDFPipeline:
    def __init__(self, config: Config):
        self.config = config
        genai.Client(api_key=self.config.api_key)
        if os.path.exists(self.config.output_file):
            os.remove(self.config.output_file)
        self.client = genai.Client(api_key=self.config.api_key)

    def clean_response(self, response_text: str):
        invalid_ctrl_pattern = r'[\x00-\x08\x0B\x0C\x0E-\x1F]'
        cleaned_text = re.sub(invalid_ctrl_pattern, '', response_text)
        return cleaned_text

    def _write_single_line(self, content: str, output_file: str):
        """Atomic write for individual JSON lines into the given output_file."""
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(content + "\n")

    def process_pdf(self, pdf_path: str):
        chapters = split_pdf_by_chapter(pdf_path, self.config)
        logging.info(f"Total {len(chapters)} chapter/subchapter PDF(s) created in '{self.config.chapters_dir}' directory.")

        # Process chapters in parallel (batching up to 4 at a time).
        for i in range(0, len(chapters), 4):
            pair = chapters[i:i+4]
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._call_llm, chapter_pdf_path, chapter_index): (chapter_index, chapter_pdf_path)
                    for chapter_index, chapter_pdf_path in pair
                }
                for future in futures:
                    chapter_index, chapter_pdf_path = futures[future]
                    try:
                        response_text = future.result()
                        self._write_jsonl_record(chapter_index, chapter_pdf_path, response_text)
                    except Exception as e:
                        logging.error(f"Error processing chapter {chapter_index}: {e}")

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
                if response.text is None:
                    raise Exception("LLM response.text is None.")
                tokens_used = (response.usage_metadata.candidates_token_count or 0) + (response.usage_metadata.total_token_count or 0)
                logging.info(f"Chapter {chapter_index}: {tokens_used} tokens")
                return response.text
            except Exception as e:
                logging.warning(f"LLM call failed for chapter {chapter_index}, attempt {attempt}/{self.config.max_retries}: {e}")
                time.sleep(2 ** attempt)
        logging.error(f"Giving up on chapter {chapter_index} after {self.config.max_retries} retries.")
        return ""

    def parse_json(self, response_text: str) -> dict:
        if response_text is None:
            raise ValueError("Response text is None.")
        cleaned = response_text.replace('```json', '').replace('```', '').strip()
        cleaned = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', cleaned)
        json_start = cleaned.find('{')
        json_end = cleaned.rfind('}') + 1
        if json_start != -1 and json_end != 0:
            cleaned = cleaned[json_start:json_end]
        cleaned = re.sub(r'}\s*{', r'},{', cleaned)
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        repaired = repair_json(cleaned)
        try:
            return json.loads(repaired)
        except Exception as e:
            raise Exception(f"Failed to parse JSON after repair: {e}\nRepaired JSON (first 300 chars): {repaired[:300]}")

    def _write_jsonl_record(self, chunk_index: tuple, chunk_pdf_path: str, response_text: str):
        """
        Writes JSON lines into a chapter-specific folder.
        `chunk_index` is a tuple: (chapter_title, subchapter_title)
        """
        chapter_title = chunk_index[0] if isinstance(chunk_index, tuple) else chunk_index
        safe_chapter = sanitize_filename(chapter_title.replace(" ", "_"))
        chapter_folder = os.path.join(self.config.chapters_dir, safe_chapter)
        os.makedirs(chapter_folder, exist_ok=True)
        
        if isinstance(chunk_index, tuple) and len(chunk_index) > 1 and chunk_index[1]:
            safe_sub = sanitize_filename(chunk_index[1].replace(" ", "_"))
            output_file = os.path.join(chapter_folder, f"{safe_sub}.jsonl")
        else:
            output_file = os.path.join(chapter_folder, f"{safe_chapter}.jsonl")
        
        try:
            response_data = self.parse_json(response_text)
            if isinstance(response_data, list):
                combined = {"qa_pairs": [], "instructional_notes": []}
                for item in response_data:
                    if isinstance(item, dict):
                        if "qa_pairs" in item and isinstance(item["qa_pairs"], list):
                            combined["qa_pairs"].extend(item["qa_pairs"])
                        if "instructional_notes" in item and isinstance(item["instructional_notes"], list):
                            combined["instructional_notes"].extend(item["instructional_notes"])
                response_data = combined
            if not isinstance(response_data, dict):
                raise Exception("Parsed JSON is not a dictionary.")
            
            for qa in response_data.get("qa_pairs", []):
                if not isinstance(qa, dict):
                    continue
                entry = {
                    "instruction": qa.get("question", "").strip(),
                    "input": "",
                    "output": qa.get("answer", "").strip(),
                    "metadata": {
                        "chapter": chapter_title,
                        "subchapter": chunk_index[1] if (isinstance(chunk_index, tuple) and len(chunk_index) > 1) else None,
                        "source": chunk_pdf_path,
                        "type": "qa"
                    }
                }
                self._write_single_line(json.dumps(entry, ensure_ascii=False), output_file)
            
            for note_idx, note in enumerate(response_data.get("instructional_notes", [])):
                entry = {
                    "instruction": note.strip(),
                    "input": "",
                    "output": "",
                    "metadata": {
                        "chapter": chapter_title,
                        "subchapter": chunk_index[1] if (isinstance(chunk_index, tuple) and len(chunk_index) > 1) else None,
                        "source": chunk_pdf_path,
                        "type": "note",
                        "note_index": note_idx + 1
                    }
                }
                self._write_single_line(json.dumps(entry, ensure_ascii=False), output_file)
        except Exception as e:
            error_entry = {
                "instruction": "UNHANDLED PROCESSING ERROR",
                "input": "",
                "output": str(e),
                "metadata": {
                    "source": os.path.basename(chunk_pdf_path),
                    "type": "error",
                    "chunk_index": chunk_index
                }
            }
            self._write_single_line(json.dumps(error_entry, ensure_ascii=False), output_file)

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
    api_key = os.getenv("api_key")
    parser = argparse.ArgumentParser(description="Split a PDF using its TOC, process via LLM, and save JSONL results into chapter-specific folders.")
    parser.add_argument("pdf_file", help="Path to the input PDF.")
    parser.add_argument("-api-key", default=api_key, help="Gemini API key (or set GEMINI_API_KEY env var).")
    parser.add_argument("--model-name", default="gemini-2.0-flash-thinking-exp-01-21", help="Gemini model name.")
    parser.add_argument("--output-file", default="pdf_chunks_output.jsonl", help="(Not used for JSON output now.)")
    parser.add_argument("--max-bytes-per-chunk", type=int, default=10_000, help="Approx. max number of bytes per PDF chunk.")
    parser.add_argument("--page-overlap", type=int, default=1, help="How many pages of overlap between consecutive chunks.")
    parser.add_argument("--max-retries", type=int, default=3, help="Number of times to retry the LLM call if it fails.")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config = Config(args)

    if not os.path.isfile(args.pdf_file):
        logging.error(f"PDF file not found: {args.pdf_file}")
        return

    pipeline = PDFPipeline(config)
    pipeline.process_pdf(args.pdf_file)

if __name__ == "__main__":
    main()

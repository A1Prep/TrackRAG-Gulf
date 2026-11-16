
"""Gemini model client for evidence-grounded question answering."""

import base64
import json
import mimetypes
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai


load_dotenv()


class GeminiAnswerModel:
    """Generate evidence-grounded answers using Gemini."""

    def __init__(self, model_name: str | None = None):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Add it to your .env file."
            )

        self.model_name = (
            model_name
            or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        )

        self.client = genai.Client(api_key=api_key)

    def generate(self, question: str, evidence: list[dict]) -> dict:
        """Send privacy-processed evidence to Gemini and parse its answer."""

        evidence_ids = [
            item["evidence_id"] for item in evidence
        ]

        prompt = f"""
You are answering a question about traffic footage.

QUESTION:
{question}

AVAILABLE EVIDENCE IDs:
{evidence_ids}

Instructions:
1. Answer only using information visible in the supplied images.
2. Do not invent details or assume facts that are not visible.
3. If the images do not establish the answer, say so.
4. Cite only evidence IDs from the available evidence list.
5. Return one JSON object with exactly these fields:
   - answer: a string
   - cited_evidence_ids: an array of evidence ID strings

Do not include Markdown fences or explanatory text outside the JSON.

Example:
{{
    "answer": "A pedestrian is visible.",
    "cited_evidence_ids": ["E1"]
}}
"""

        inputs = [{"type": "text", "text": prompt}]

        for item in evidence:
            evidence_id = item["evidence_id"]

            for image in item["images"]:
                image_path = Path(image["image_path"])

                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Processed evidence image not found: {image_path}"
                    )

                mime_type, _ = mimetypes.guess_type(str(image_path))

                if not mime_type or not mime_type.startswith("image/"):
                    raise ValueError(
                        f"Unsupported evidence image type: {image_path}"
                    )

                encoded_image = base64.b64encode(
                    image_path.read_bytes()
                ).decode("utf-8")

                inputs.append({
                    "type": "text",
                    "text": (
                        f"Evidence ID: {evidence_id}. "
                        f"Frame ID: {image['frame_id']}."
                    ),
                })

                inputs.append({
                    "type": "image",
                    "data": encoded_image,
                    "mime_type": mime_type,
                })

        interaction = self.client.interactions.create(
            model=self.model_name,
            input=inputs,
        )

        response_text = (interaction.output_text or "").strip()

        print("\n--- RAW GEMINI RESPONSE ---")
        print(repr(response_text[:1500]))
        print("--- END RAW RESPONSE ---\n")

        if not response_text:
            raise ValueError("Gemini returned an empty response.")

        # Remove Markdown JSON fences if Gemini included them.
        cleaned_text = re.sub(
            r"^\s*```(?:json)?\s*|\s*```\s*$",
            "",
            response_text,
            flags=re.IGNORECASE,
        ).strip()

        # Extract the JSON object if additional text surrounds it.
        if not cleaned_text.startswith("{"):
            start = cleaned_text.find("{")
            end = cleaned_text.rfind("}")

            if start != -1 and end > start:
                cleaned_text = cleaned_text[start:end + 1]

        try:
            result = json.loads(cleaned_text)
        except json.JSONDecodeError as error:
            raise ValueError(
                "Gemini did not return valid JSON. "
                f"Response: {response_text[:1000]!r}"
            ) from error

        if not isinstance(result, dict):
            raise ValueError("Gemini response must be a JSON object.")

        if "answer" not in result or "cited_evidence_ids" not in result:
            raise ValueError(
                "Gemini response is missing required fields."
            )

        return result

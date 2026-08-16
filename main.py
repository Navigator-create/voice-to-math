"""Local mock Math Voice Verifier demo.

The uploaded audio is intentionally ignored.  The endpoint returns a fixed
transcript so the browser upload flow can be exercised without Whisper or any
external service.
"""

import ast
import logging
import math
import os
from pathlib import Path
import re

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
import uvicorn


BASE_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 1_000_000
UNRECOGNIZABLE_RESULT = "N/A (Not a recognizable math expression)"
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}
SPOKEN_OPERATORS = {
    "to the power of": "**",
    "multiplied by": "*",
    "divided by": "/",
    "times": "*",
    "plus": "+",
    "minus": "-",
}

logger = logging.getLogger(__name__)
app = FastAPI(title="Math Voice Verifier (Mock)")
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")


class UnsafeExpression(ValueError):
    """Raised when an expression is outside the deliberately small grammar."""


class TranscriptionUnavailable(RuntimeError):
    """Raised when the configured transcription service cannot provide text."""


def _evaluate_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UnsafeExpression("Only numeric literals are allowed")
        if not math.isfinite(node.value):
            raise UnsafeExpression("Numbers must be finite")
        return node.value

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value

    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)

        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            result = left / right
        elif isinstance(node.op, ast.Pow):
            if abs(right) > 1_000 or abs(left) > 1_000_000:
                raise UnsafeExpression("Exponentiation is too large")
            result = left**right
        else:
            raise UnsafeExpression("Operator is not allowed")

        if isinstance(result, complex) or not math.isfinite(result):
            raise UnsafeExpression("Result must be a finite real number")
        return result

    raise UnsafeExpression("Expression contains unsupported syntax")


def evaluate_arithmetic(expression: str) -> str:
    """Evaluate a limited arithmetic expression without evaluating Python code."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate_node(tree.body)
    except (SyntaxError, UnsafeExpression, ZeroDivisionError, OverflowError, ValueError):
        return UNRECOGNIZABLE_RESULT

    return str(result)


def normalize_spoken_math(transcription: str) -> str:
    """Convert a deliberately small English maths vocabulary to symbols."""
    expression = transcription.strip().lower().rstrip(".!?")
    if not expression:
        return expression

    for phrase, operator in SPOKEN_OPERATORS.items():
        expression = re.sub(rf"\b{re.escape(phrase)}\b", f" {operator} ", expression)
    for word, numeral in NUMBER_WORDS.items():
        expression = re.sub(rf"\b{word}\b", numeral, expression)

    # Unknown words are deliberately left untouched so the safe evaluator rejects
    # them rather than guessing at language outside the supported vocabulary.
    return re.sub(r"\s+", " ", expression).strip()


def transcription_settings() -> tuple[str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_TRANSCRIPTION_MODEL")
    if not api_key or not model:
        raise HTTPException(
            status_code=503,
            detail="OpenAI transcription is not configured on this server",
        )
    return api_key, model


async def transcribe_audio(audio: bytes, filename: str, content_type: str) -> str:
    """Send one validated recording to OpenAI and return its plain transcript."""
    api_key, model = transcription_settings()
    client = AsyncOpenAI(api_key=api_key)
    try:
        transcript = await client.audio.transcriptions.create(
            model=model,
            file=(filename, audio, content_type),
            language="en",
            response_format="json",
        )
        text = getattr(transcript, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("OpenAI returned an empty transcription")
        return text.strip()
    except Exception as error:
        raise TranscriptionUnavailable("OpenAI transcription request failed") from error
    finally:
        await client.close()


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html")


@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)) -> JSONResponse:
    try:
        if not file.content_type or not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=415, detail="Upload an audio file")

        audio = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(audio) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Audio files must be {MAX_UPLOAD_BYTES // 1_000_000} MB or smaller",
            )

        transcription = await transcribe_audio(
            audio,
            file.filename or "recording.webm",
            file.content_type,
        )
        return JSONResponse(
            {
                "transcription": transcription,
                "result": evaluate_arithmetic(normalize_spoken_math(transcription)),
                "mode": "openai",
            }
        )
    except HTTPException:
        raise
    except TranscriptionUnavailable:
        logger.exception("OpenAI transcription request failed")
        return JSONResponse(
            status_code=502,
            content={"detail": "Transcription service is unavailable"},
        )
    except Exception:
        logger.exception("Unable to process uploaded audio")
        return JSONResponse(status_code=500, content={"detail": "Unable to process audio"})
    finally:
        await file.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

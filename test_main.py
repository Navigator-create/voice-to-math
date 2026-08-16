"""Focused offline checks for the local OpenAI transcription integration."""

import asyncio
import io
import json
import os
from tempfile import SpooledTemporaryFile
import unittest

from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile

import main


class FakeTranscription:
    text = "six times seven"


class FakeTranscriptions:
    def __init__(self, owner):
        self.owner = owner

    async def create(self, **kwargs):
        self.owner.request = kwargs
        return FakeTranscription()


class FakeOpenAI:
    instance = None

    def __init__(self, *, api_key):
        self.api_key = api_key
        self.audio = type("Audio", (), {"transcriptions": FakeTranscriptions(self)})()
        self.request = None
        self.closed = False
        FakeOpenAI.instance = self

    async def close(self):
        self.closed = True


def audio_upload(name: str, body: bytes, content_type: str) -> UploadFile:
    storage = SpooledTemporaryFile(max_size=main.MAX_UPLOAD_BYTES + 2)
    storage.write(body)
    storage.seek(0)
    return UploadFile(
        file=storage,
        filename=name,
        headers=Headers({"content-type": content_type}),
    )


class MathVoiceTests(unittest.TestCase):
    def setUp(self):
        self.original_client = main.AsyncOpenAI
        main.AsyncOpenAI = FakeOpenAI
        self.original_env = dict(os.environ)
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["OPENAI_TRANSCRIPTION_MODEL"] = "test-model"

    def tearDown(self):
        main.AsyncOpenAI = self.original_client
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_normalizes_supported_spoken_math(self):
        self.assertEqual(main.evaluate_arithmetic(main.normalize_spoken_math("two plus two")), "4")
        self.assertEqual(main.evaluate_arithmetic(main.normalize_spoken_math("six times seven")), "42")
        self.assertEqual(main.evaluate_arithmetic(main.normalize_spoken_math("two to the power of three")), "8")
        self.assertEqual(
            main.evaluate_arithmetic(main.normalize_spoken_math("please add two and two")),
            main.UNRECOGNIZABLE_RESULT,
        )

    def test_transcription_request_and_response(self):
        response = asyncio.run(main.process_audio(audio_upload("recording.webm", b"audio", "audio/webm")))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body), {
            "transcription": "six times seven",
            "result": "42",
            "mode": "openai",
        })
        self.assertEqual(FakeOpenAI.instance.request["model"], "test-model")
        self.assertEqual(FakeOpenAI.instance.request["file"][0], "recording.webm")
        self.assertTrue(FakeOpenAI.instance.closed)

    def test_missing_configuration_is_controlled(self):
        os.environ.pop("OPENAI_API_KEY")
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(main.process_audio(audio_upload("recording.webm", b"audio", "audio/webm")))
        self.assertEqual(raised.exception.status_code, 503)

    def test_invalid_and_oversized_uploads_are_rejected(self):
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(main.process_audio(audio_upload("notes.txt", b"text", "text/plain")))
        self.assertEqual(raised.exception.status_code, 415)

        with self.assertRaises(HTTPException) as raised:
            asyncio.run(
                main.process_audio(
                    audio_upload("large.webm", b"x" * (main.MAX_UPLOAD_BYTES + 1), "audio/webm")
                )
            )
        self.assertEqual(raised.exception.status_code, 413)


if __name__ == "__main__":
    unittest.main()

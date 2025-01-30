from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel
import torch
from io import BytesIO

app = FastAPI()

# Load the Whisper model
MODEL_NAME = "ivrit-ai/faster-whisper-v2-d4"
model = WhisperModel(
    MODEL_NAME,
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="int8",
)


@app.websocket("/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive raw WebM data from client
            audio_bytes = await websocket.receive_bytes()

            # Convert bytes to file-like object
            audio_file = BytesIO(audio_bytes)

            # Transcribe the audio using Whisper
            segments, _ = model.transcribe(audio_file, language="he")
            transcription = " ".join([segment.text for segment in segments])

            # Send the transcription back to the client
            await websocket.send_text(transcription)

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8121)

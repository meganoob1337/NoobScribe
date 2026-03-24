# From https://github.com/jfgonsalves/parakeet-diarized (commit 6abadfd)
# Copyright (c) jfgonsalves - MIT License
# String updates for NoobScribe by meganoob1337
import os
import logging
import uvicorn
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Enable debug logging if requested
if os.environ.get('DEBUG', '0') == '1':
    logger.setLevel(logging.DEBUG)
    logger.debug("Debug logging enabled")

# Import from modularized components
from api import create_app
from config import get_config

# Get the configuration
config = get_config()

# Create the FastAPI application
app = create_app()

# For backwards compatibility - re-export the split_audio_into_chunks function
# This is needed for the test_chunking.py to work without modification
from audio import split_audio_into_chunks, convert_audio_to_wav
from transcription import load_model, format_srt, format_vtt
from models import WhisperSegment, TranscriptionResponse

# Run the server if executed directly
if __name__ == "__main__":
    # Log startup information
    logger.info(f"Starting NoobScribe API on {config.host}:{config.port}")
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available, using CPU (this will be slow)")
    
    # Start the server
    uvicorn.run(app, host=config.host, port=config.port, reload=config.debug)
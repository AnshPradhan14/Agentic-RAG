import nltk
import tiktoken
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Testing NLTK download...")
try:
    nltk.download("punkt", quiet=False)
    nltk.download("punkt_tab", quiet=False)
    logger.info("NLTK download success.")
except Exception as e:
    logger.error(f"NLTK download failed: {e}")

logger.info("Testing tiktoken encoding load...")
try:
    enc = tiktoken.get_encoding("cl100k_base")
    logger.info("tiktoken encoding load success.")
except Exception as e:
    logger.error(f"tiktoken encoding load failed: {e}")

logger.info("Test complete.")

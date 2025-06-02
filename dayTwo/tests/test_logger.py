"""Tests for the logger utility module."""

import pytest
import logging
from pathlib import Path
from src.utils.logger import setup_logger

def test_logger_initialization():
    """Test logger initialization."""
    logger = setup_logger('test_logger')
    assert isinstance(logger, logging.Logger)
    assert logger.name == 'test_logger'
    assert logger.level == logging.INFO

def test_logger_handlers():
    """Test logger handlers setup."""
    logger = setup_logger('test_logger')
    assert len(logger.handlers) == 1  # Only console handler by default
    assert isinstance(logger.handlers[0], logging.StreamHandler)

def test_logger_file_handler(tmp_path):
    """Test logger with file handler."""
    log_file = tmp_path / "test.log"
    logger = setup_logger('test_logger', log_file=log_file)
    
    assert len(logger.handlers) == 2  # Console and file handlers
    assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    
    # Test logging to file
    test_message = "Test log message"
    logger.info(test_message)
    
    assert log_file.exists()
    with open(log_file, 'r') as f:
        log_content = f.read()
        assert test_message in log_content

def test_logger_custom_level():
    """Test logger with custom level."""
    logger = setup_logger('test_logger', level=logging.DEBUG)
    assert logger.level == logging.DEBUG

def test_logger_format(caplog):
    """Test logger message format."""
    caplog.set_level(logging.INFO)
    logger = setup_logger('test_logger')
    test_message = "Test message"
    
    logger.info(test_message)
    
    # Check if the log message contains all required components
    assert test_message in caplog.text
    assert 'test_logger' in caplog.text
    assert 'INFO' in caplog.text 
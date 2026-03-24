# From https://github.com/jfgonsalves/parakeet-diarized (commit 6abadfd)
# Copyright (c) jfgonsalves - MIT License
import os
import tempfile
import unittest
import subprocess
from unittest.mock import patch, MagicMock
import wave
import numpy as np

# Import the function to test
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from audio import split_audio_into_chunks


class TestAudioChunking(unittest.TestCase):
    def create_test_wav(self, duration_seconds=10, sample_rate=16000):
        """Create a test WAV file with the specified duration"""
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file.close()
        
        # Generate sample audio data (sine wave)
        t = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate), False)
        data = np.sin(2 * np.pi * 440 * t) * 32767
        data = data.astype(np.int16)
        
        # Write to WAV file
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(data.tobytes())
            
        return temp_file.name
    
    def test_no_chunking_for_short_audio(self):
        """Test that short audio files are not chunked"""
        # Create a 4-second test WAV file (under the 5-minute threshold)
        test_file = self.create_test_wav(duration_seconds=4)
        try:
            # Call the function with a 5-minute (300 second) chunk size
            result = split_audio_into_chunks(test_file, chunk_duration=300)
            
            # Should return a list with just the original file
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], test_file)
        finally:
            # Clean up
            if os.path.exists(test_file):
                os.unlink(test_file)
    
    @patch('subprocess.run')
    def test_chunking_for_long_audio(self, mock_subprocess_run):
        """Test that long audio files are properly chunked"""
        # Setup the mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stderr = b''
        mock_subprocess_run.return_value = mock_process
        
        # Create a test WAV file
        test_file = self.create_test_wav(duration_seconds=600)  # 10 minutes
        
        try:
            # Mock wave.open to make it return our desired duration
            original_wave_open = wave.open
            
            def mock_wave_open(file, mode):
                wav_file = original_wave_open(file, mode)
                if file == test_file and mode == 'rb':
                    # Override the getnframes and getframerate methods to fake a 10-minute file
                    original_getnframes = wav_file.getnframes
                    original_getframerate = wav_file.getframerate
                    wav_file.getnframes = lambda: 16000 * 600  # 10 minutes at 16kHz
                    wav_file.getframerate = lambda: 16000
                return wav_file
                
            # Apply the mock
            with patch('wave.open', mock_wave_open):
                # Call the function with a 5-minute (300 second) chunk size
                result = split_audio_into_chunks(test_file, chunk_duration=300)
                
                # Should return 2 chunks for a 10-minute file with 5-minute chunks
                self.assertEqual(len(result), 2)
                
                # Check ffmpeg was called twice (once for each chunk)
                self.assertEqual(mock_subprocess_run.call_count, 2)
                
                # Check ffmpeg parameters
                first_call_args = mock_subprocess_run.call_args_list[0][0][0]
                self.assertIn('-ss', first_call_args)
                self.assertIn('0', first_call_args[first_call_args.index('-ss') + 1])  # First chunk starts at 0
                
                second_call_args = mock_subprocess_run.call_args_list[1][0][0]
                self.assertIn('-ss', second_call_args)
                self.assertIn('300', second_call_args[second_call_args.index('-ss') + 1])  # Second chunk starts at 300s
        finally:
            # Clean up
            if os.path.exists(test_file):
                os.unlink(test_file)


if __name__ == '__main__':
    unittest.main()
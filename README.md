# Resume-Parser
A 3rd-Party Python LLM-based API for Resume Parser.

## Ollama Setup

To set up Ollama on your system, use the provided `run.sh` script. This script will install Ollama (if not already installed) and start the Ollama service.

### Steps:

1. Open a terminal in the project directory.
2. Make the script executable:
   ```sh
   chmod +x run.sh
   ```
3. Run the setup script:
   ```sh
   ./run.sh
   ```

## Testing the Setup

To test the Ollama API integration, use the `test.py` script. It sends a prompt to the Ollama server and prints the response.

### Steps:

1. Create a Python virtual environment:
   ```sh
   python -m venv venv
   ```
2. Activate the virtual environment:
   - On Linux/macOS:
     ```sh
     source venv/bin/activate
     ```
   - On Windows:
     ```sh
     .\venv\Scripts\activate
     ```
3. Install the required dependencies:
   ```sh
   pip install requests
   ```
4. Run the test script:
   ```sh
   python test.py
   ```

---

For PDF extraction and resume parsing, see the `Extractor/` directory for more details.

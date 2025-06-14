# Resume Parser API

This project provides a FastAPI-based API for parsing resumes from PDF files, extracting structured information, and validating user identity. It supports advanced PDF extraction, segmentation, and structuring logic.

## 1. Clone the Repository

```bash
# On your VM, clone the repository
git clone https://github.com/Nexio-Developer-Group/Resume-Parser.git
cd Resume-Parser
```

## 2. Install Python venv (if not already installed)

```bash
sudo apt install python3 python3-venv python3-pip -y
```

## 3. Create and Activate a Virtual Environment

```bash
# In the root of the repository
python3 -m venv venv
source venv/bin/activate
```

## 4. Install Project Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 5. Set Up Environment Variables

Create a `.env` file in the root directory with your API key:

```
API_KEY=your_secret_api_key_here
```

## 6. Run the FastAPI Application

```bash
# From the root directory, with the venv activated
uvicorn app.main:app --host 127.0.0.1 --port 8000
# use this command if want to run the script without activating venv
./venv/bin/python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- The API will be available at: http://<your-vm-ip>:8000/
- The interactive docs will be at: http://<your-vm-ip>:8000/docs

## 7. (Optional) Run the API with PM2 for Production

[PM2](https://pm2.keymetrics.io/) is a process manager for Node.js, but it can also manage Python processes.

### Install PM2 (with Node.js)

```bash
sudo apt install -y nodejs npm
sudo npm install -g pm2
```

### Run Uvicorn with PM2

```bash
# From the root directory, with the venv activated
pm2 start ./venv/bin/python --name resume-parser --   -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# To see logs
pm2 logs resume-api

# To restart/stop
pm2 restart resume-api
pm2 stop resume-api
```

---

## License

This project is licensed under the MIT License.

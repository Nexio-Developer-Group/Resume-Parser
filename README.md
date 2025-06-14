# Resume Parser API

This project provides a FastAPI-based API for parsing resumes from PDF files, extracting structured information, and validating user identity. It supports advanced PDF extraction, segmentation, and structuring logic.

## 1. Clone the Repository

```bash
# On your VM, clone the repository
git clone https://github.com/Nexio-Developer-Group/Resume-Parser.git
cd Resume-Parser
```

## 2. Install Python 3.10.11 on Ubuntu

```bash
sudo apt update
sudo apt install -y wget build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl libncursesw5-dev \
    xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Download and install Python 3.10.11
wget https://www.python.org/ftp/python/3.10.11/Python-3.10.11.tgz
tar -xf Python-3.10.11.tgz
cd Python-3.10.11
./configure --enable-optimizations
make -j $(nproc)
sudo make altinstall

# Verify installation
python3.10 --version
cd ..
```

## 3. Install Python venv (if not already installed)

```bash
sudo apt install -y python3.10-venv
```

## 4. Create and Activate a Virtual Environment

```bash
# In the root of the repository
python3.10 -m venv venv
source venv/bin/activate
```

## 5. Install Project Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Set Up Environment Variables

Create a `.env` file in the root directory with your API key:

```
API_KEY=your_secret_api_key_here
```

## 8. Run the FastAPI Application

```bash
# From the root directory, with the venv activated
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- The API will be available at: http://<your-vm-ip>:8000/
- The interactive docs will be at: http://<your-vm-ip>:8000/docs

## 9. (Optional) Run the API with PM2 for Production

[PM2](https://pm2.keymetrics.io/) is a process manager for Node.js, but it can also manage Python processes.

### Install PM2 (with Node.js)

```bash
sudo apt install -y nodejs npm
sudo npm install -g pm2
```

### Run Uvicorn with PM2

```bash
# From the root directory, with the venv activated
pm2 start venv/bin/uvicorn --name resume-api -- app.main:app --host 0.0.0.0 --port 8000

# To see logs
pm2 logs resume-api

# To restart/stop
pm2 restart resume-api
pm2 stop resume-api
```

---

## License

This project is licensed under the MIT License.

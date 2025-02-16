# Loan Defaulter Prediction Web App

## Overview

This project is a Loan Defaulter Prediction web application that leverages machine learning and LLM integration. It utilizes the "llama3-8b-8192" model provided by Groq for API requests.

The ML model used in this project is a **Random Forest model**. Its training and preprocessing details can be found at: [Loan Defaulter Prediction and LLM Report](https://github.com/cdacPrj/LoanDefaulterPrediction_and_LLM_report).

## Steps to Run the Web App

### 1. Clone Git Repository

```sh
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Create a New Python Virtual Environment

```sh
python -m venv myenv
```

### 3. Activate Virtual Environment

#### On Windows:

```sh
myenv\Scripts\activate
```

#### On macOS/Linux:

```sh
source myenv/bin/activate
```

### 4. Install Dependencies

```sh
pip install -r requirements.txt
```

### 5. Create a Configuration File

Create a new file named `config.ini` in the project root directory with the following structure:

```
[groq]
GROQ_API_KEY = gsk_****************************************
```

This file contains the API key for accessing the "llama3-8b-8192" model provided by Groq.

### 6. Run the Project

```sh
python app.py
```

## Additional Requirements

- The project requires an **active internet connection** to access the Groq API.

## Contributors

- **Kaustubh Vanalkar**
- **Rushank Wani**



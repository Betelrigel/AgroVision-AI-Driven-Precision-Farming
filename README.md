**Crop Disease Detection App**

- Rice Dataset → The app can detect four major rice diseases:
  Bacterial Blight, Blast Disease, Brown Spot, and False Smut.
- Wheat Dataset → For wheat, the app classifies yellow rust–affected leaves into different resistance/severity levels (0, MR, MRMS, MS, R, S).

Datasets
<img width="399" height="372" alt="image" src="https://github.com/user-attachments/assets/10c5d26d-cbd7-4efb-ae09-069504ffc55f" />

The models are trained using the following public datasets:

- Yellow Rust in Wheat
  https://www.kaggle.com/datasets/tolgahayit/yellowrust19-yellow-rust-disease-in-wheat

- Rice Crop Diseases
  https://www.kaggle.com/datasets/thegoanpanda/rice-crop-diseases

---

Installation & Setup

1.  Clone the repository

        git clone https://github.com/yourusername/crop-disease-detection.git
        cd crop-disease-detection

2.  Create a virtual environment & install dependencies

        python -m venv venv
        source venv/bin/activate   # On Linux/Mac
        venv\Scripts\activate      # On Windows
        pip install -r requirements.txt

3.  Set up OpenWeather API key

    - Create a free account on OpenWeather.

    - Get your API key.

    - Create a .env file in the project root and add:

          OPENWEATHER_API_KEY=your_api_key_here

4.  Run the Application

        python app.py

    The app will start on http://127.0.0.1:5000.

---

Model Execution

- Wheat Yellow Rust Model → Detects yellow rust in wheat leaves.
- Rice Disease Model → Identifies multiple rice crop diseases.

The models are pre-trained and loaded automatically.

---

LLM Integration

For conversational insights (e.g., advisory support), you can integrate
an LLM.

Option A: Run Locally with Ollama

- Install Ollama.

- Pull and run a model (e.g., Llama 3 or Caly):

      ollama pull llama3
      ollama run llama3

Option B: Use an API (Groq, OpenAI, etc.)

- Update the code to call your preferred API.

- Set the API key in .env:

      OPENAI_API_KEY=your_key_here
      GROQ_API_KEY=your_key_here

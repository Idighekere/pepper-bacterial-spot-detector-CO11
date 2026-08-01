# Pepper Bacterial Spot Detector — CO11

A binary image classifier that distinguishes between **healthy pepper leaves** and **pepper leaves infected with Bacterial Spot**, built with TensorFlow and deployed as a Streamlit web application with a custom neobrutalist UI.

## Task

| Group | Classification Task                     |
| ----- | --------------------------------------- |
| CO11  | Pepper Healthy vs Pepper Bacterial Spot |

## Dataset

Download from Kaggle: [Pepper Belly Crop (PlantVillage DS)](https://www.kaggle.com/datasets/zienabesam/pepper-belly-crop-plantvillage-ds)

Only the **Pepper** classes are used:

- `Pepper,_bell___Bacterial_spot` — diseased
- `Pepper,_bell___healthy` — healthy

**Note:** The dataset is not pushed to the repository (see `.gitignore`).

To set it up locally:
1. Download from Kaggle and unzip into `datasets/`.
2. Run the data-prep cell in `notebooks/train_model.ipynb` — it maps the raw Kaggle folder names to clean names and creates the `train/val/test` splits.

This produces:

```
datasets/
├── train/
│   ├── Bacterial_Spot/
│   └── Healthy/
├── val/
│   ├── Bacterial_Spot/
│   └── Healthy/
└── test/
    ├── Bacterial_Spot/
    └── Healthy/
```

## How It Works

1. Upload a photo of a pepper leaf through the web interface.
2. The image is resized to 128×128 and checked against a gallery of known leaf embeddings.
3. If the photo does not resemble any training leaf, the app returns a "Not a Leaf" notice.
4. Otherwise the model returns a Healthy / Bacterial Spot prediction with a confidence score.
5. The result is displayed on the page.

## How to Run Locally

### Prerequisites

- Python 3.x
- Git
- Kaggle account (to download the dataset)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Idighekere/pepper-bacterial-spot-detector-CO11.git
cd pepper-bacterial-spot-detector-CO11

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Set up the dataset and train the model (optional — a pre-trained model is included)
#    Download from Kaggle and unzip into datasets/, then open and run
#    notebooks/train_model.ipynb (it splits the data and trains the model)

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Deployed App

Deployed on Streamlit Community Cloud:

[Live Demo](https://pepper-bacterial-spot-detector-co11.streamlit.app)

### Deploying to Streamlit Cloud

1. Push the repository to GitHub.
2. Go to [streamlit.io](https://streamlit.io) and sign in with GitHub.
3. Click **Create app**, select the repo and branch, and set the main file to `app.py`.
4. Streamlit auto-installs `requirements.txt` and serves the app.

## Tech Stack

- **Model**: TensorFlow / Keras (Custom CNN)
- **UI**: Streamlit + custom CSS (neobrutalist)
- **Deployment**: Streamlit Community Cloud
- **Version Control**: Git & GitHub

## Project Structure

```
├── app.py                  # Streamlit application
├── flask_app.py            # Legacy Flask app (kept as backup)
├── build_reference.py      # Builds the leaf-embedding gallery (Not-a-Leaf guard)
├── models/
│   ├── custom_cnn_best.keras  # Trained model
│   ├── custom_cnn.keras
│   └── leaf_reference.npz  # Embedding gallery + threshold for the guard
├── templates/
│   └── index.html          # Legacy Flask template (backup)
├── notebooks/
│   └── train_model.ipynb   # Data split + model training notebook
├── .streamlit/
│   └── config.toml         # Streamlit theme/server config
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

## Challenges & Improvements

1. **False sick reports** — every leaf was reported as Bacterial Spot at ~66% confidence. The model already rescales pixel values internally, but the app normalized them a second time. Fixed by removing the duplicate normalization.
2. **Confident guesses on random photos** — the model always picks one of its two classes, so a non-leaf photo got a confident but meaningless result. Added a "Not a Leaf" guard that compares each photo's embedding to thousands of known leaf photos.
3. **Limited hardware resources** — no GPU and little memory made training slow and crash-prone. Training moved to Google Colab, and the model was kept small (322K parameters, 128×128 input) with TensorFlow thread limits to fit low-memory hosting.
4. **Deployment** — the initial Render deployment struggled with memory and request timeouts. Moved to Streamlit Community Cloud, which auto-installs dependencies and serves the app.

## Contributors

| Name           | Reg. No.      | GitHub                                             |
| -------------- | ------------- | -------------------------------------------------- |
| Idighekere Udo | 22/EG/CO/1715 | [@idighekere](https://github.com/idighekere)       |
| Ekemini-Abasi Tom      | 22/EG/CO/1655 | [@tomrex22](https://github.com/tomrex22)   |
| Bassey, Kuyik      | 22/EG/CO/1775 | [@iamkuyik](https://github.com/iamkuyik) |
| Bankole John .O   | 22/EG/CO/1725 | [@Johnbo616](https://github.com/Johnbo616) |
| Udofia, Aniebietabasi A.      | 22/EG/CO/1765 | [@aniebietabasi01](https://github.com/aniebietabasi01) |
| Okposin, Edidiong      | 22/EG/CO/1635 | [@22EGCO1635](https://github.com/22EGCO1635) |
| Eno, Abasiono  | 22/EG/CO/1745 | [@Gemspixelz](https://github.com/Gemspixelz) |
| Bassey, Abasiama I.  | 22/EG/CO/1755 | [@abasiamainemesit](https://github.com/abasiamainemesit) |
| Okure, Praise Okure    | 22/EG/CO/1735 | [@Praiz05](https://github.com/Praiz05) |
 | Okereke, Arizonachi Lynn    | 22/EG/CO/1785 | [@Zonarh](https://github.com/Zonarh) |
 | Edet, Abasiama Sunday    | 22/EG/CO/1675 | [@Edetcode](https://github.com/Edetcode) |
 | Akpabio Martin Anthony | 22/EG/CO/1815 | [@rextyler9](https://github.com/rextyler9) |
  | Your name | 22/EG/CO/XXXX | [@username](https://github.com/username) |

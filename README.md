# Pepper Bacterial Spot Detector — CO11

A binary image classifier that distinguishes between **healthy pepper leaves** and **pepper leaves infected with Bacterial Spot**, built with TensorFlow and deployed as a Flask web application.

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
To set it up locally, run the provided split script after downloading:

```bash
# 1. Download the dataset from Kaggle and unzip it
#    into datasets/Pepper Belly Crop DS/

# 2. Run the split script (creates train/val/test splits with clean folder names)
python split_dataset.py
```

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
2. The image is preprocessed (resized to 224×224, normalized) and passed through a trained CNN / transfer learning model.
3. The model returns a prediction with a confidence score.
4. The result is displayed on the page.

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

# 3. Set up the dataset
#    - Download from Kaggle and unzip into datasets/Pepper Belly Crop DS/
#    - Then run:
python split_dataset.py

# 4. Train the model (optional — a pre-trained model is included)
#    Open and run training/train_model.ipynb in Jupyter

# 5. Install dependencies
pip install -r requirements.txt

# 6. Run the app
flask run
# or
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Deployed App

[Live Demo](https://pepper-bacterial-spot-detector-co11.onrender.com) — deployed on Render

## Tech Stack

- **Model**: TensorFlow / Keras (Custom CNN + MobileNetV3 Transfer Learning)
- **Backend**: Flask
- **Deployment**: Render (Gunicorn) / Hugging Face
- **Version Control**: Git & GitHub

## Project Structure

```
├── app.py                  # Flask application
├── models/
│   └── pepper_model.keras  # Trained model
├── templates/
│   └── index.html          # Web interface
├── training/
│   └── train_model.ipynb   # Model training notebook
├── requirements.txt        # Python dependencies
├── split_dataset.py        # Script to split raw download into train/val/test
├── .gitignore
└── README.md
```

## Challenges & Improvements

_To be completed after development._

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
Okure, Praise Okure    | 22/EG/CO/1735 | [@Praiz05](https://github.com/Praiz05) |
 | Your Name      | 22/EG/CO/XXXX | [@your-username](https://github.com/your-username) |
 | Your Name      | 22/EG/CO/XXXX | [@your-username](https://github.com/your-username) |
 | Your Name      | 22/EG/CO/XXXX | [@your-username](https://github.com/your-username) |
 
 _ Add other group members above _

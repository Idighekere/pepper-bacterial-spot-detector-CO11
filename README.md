# Pepper Bacterial Spot Detector — CO11

A binary image classifier that distinguishes between **healthy pepper leaves** and **pepper leaves infected with Bacterial Spot**, built with TensorFlow and deployed as a Flask web application.

## Task

| Group | Classification Task                |
|-------|-------------------------------------|
| CO11  | Pepper Healthy vs Pepper Bacterial Spot |

## Dataset

Source: [PlantVillage Dataset](https://www.kaggle.com/datasets/tushar5harma/plantvillage-dataset) on Kaggle.

The dataset contains leaf images of multiple crop species. Only the **Pepper** classes are used:
- `Pepper__bell___Bacterial_spot` — diseased
- `Pepper__bell___healthy` — healthy

## How It Works

1. Upload a photo of a pepper leaf through the web interface.
2. The image is preprocessed (resized to 224×224, normalized) and passed through a trained CNN / transfer learning model.
3. The model returns a prediction with a confidence score.
4. The result is displayed on the page.

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Idighekere/pepper-bacterial-spot-detector-CO11.git
cd pepper-bacterial-spot-detector-CO11

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
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
- **Deployment**: Render (Gunicorn)
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
├── .gitignore
└── README.md
```

## Challenges & Improvements

*To be completed after development.*

## Contributors

| Name | GitHub |
|------|--------|
| Idighekere Udo | [@idighekere](https://github.com/idighekere) |

*Add other group members above.*

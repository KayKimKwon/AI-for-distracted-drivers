# AI for Distracted Driver Detection

A machine learning project that classifies distracted drivers into four categories: **Attentive**, **Drinking Coffee**, **Using Mirror**, and **Using Radio**.

## Models
- **KNN** (K-Nearest Neighbors)
- **CNN** (Convolutional Neural Network)
- **VGG16** (Transfer Learning)

## Changes from Original
This project was originally built in a Jupyter Notebook during a summer camp, where most of the setup code was provided by instructors. It has since been independently developed with the following changes:

- Ported from Jupyter Notebook to run in a standard Python IDE (VS Code)
- KNN visualizations now display labels as image titles
- Users can choose how many training samples to observe
- Driver faces are censored in displayed images for privacy

## Setup
Dependencies are installed via:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

Data is downloaded automatically when running the script.

## Credits
Original project structure and setup code provided by instructors at NSLC summer camp 2024.

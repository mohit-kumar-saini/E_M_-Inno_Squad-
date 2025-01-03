# E_M_-Inno_Squad-

## Overview
E_M_-Inno_Squad- is an project that uses emotion detection to recommend music based on the user's preferences and mood. By leveraging computer vision and machine learning, this application captures the user's emotions through facial expressions and suggests songs tailored to their emotional state.

## Features
- **Emotion Detection:** Utilizes Mediapipe and a trained model to identify emotions from facial landmarks and hand gestures.
- **Music Recommendations:** Recommends songs based on detected emotions, preferred language, and favorite singer.
- **Interactive Interface:** Allows users to input preferences and interact with the camera for real-time emotion capture.
- **Seamless Integration:** Opens YouTube to play recommended songs.

## Technologies Used
- **Frontend:** Streamlit for building an interactive web application.
- **Backend:**
  - OpenCV for camera and image processing.
  - Mediapipe for facial and hand landmark detection.
  - TensorFlow/Keras for emotion classification using a pre-trained model.
- **Other Tools:**
  - NumPy for data manipulation.
  - Webbrowser module for opening YouTube links.

## Installation
Follow these steps to set up and run the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/mohit-kumar-saini/E_M_-Inno_Squad-.git
   ```

2. Navigate to the project directory:
   ```bash
   cd E_M_-Inno_Squad-
   ```

3. Install all the required dependencies
   
4. Ensure the following files are present in the project directory:
   - `model.h5`: Pre-trained model for emotion classification.
   - `labels.npy`: Array containing emotion labels.

5. Run the application:
   ```bash
   streamlit run main.py
   ```

6. Open the application in your browser:
   ```
   http://localhost:8501
   ```

## Usage
1. Launch the application and allow camera access.
2. Input your preferred language and favorite singer.
3. Start the camera to capture your emotions.
4. Click on "Recommend me songs!" to get personalized music recommendations.

## Contribution Guidelines
We welcome contributions! Follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push to your fork:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

## Contact
For any inquiries or support, please contact:

- **Name:** Mohit Kumar Saini  
- **Email:** mohitsaini2622132@gmail.com  
- **LinkedIn:** linkedin.com/in/mohit-kumar-saini-18389a250

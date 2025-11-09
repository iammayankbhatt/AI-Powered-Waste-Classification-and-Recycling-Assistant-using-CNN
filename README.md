# ♻️ AI-Powered Waste Classification using CNN & Grad-CAM

This project classifies waste images into **Organic** or **Recyclable** categories using a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras** and fine-tuned with **MobileNetV2** for high accuracy.  
It includes an interactive **Streamlit web app** that lets users upload images, view predictions, and see **Grad-CAM visualizations** highlighting the regions that influenced the model’s decision — making the AI explainable and trustworthy.

---

## 🚀 Features

- 🧠 **Deep Learning Model:** MobileNetV2 transfer learning + fine-tuning (~92% test accuracy)  
- 🔍 **Explainable AI:** Grad-CAM overlay shows what part of the image the model focused on  
- 🌐 **Streamlit Web App:** Simple UI for real-time testing and predictions  
- 📊 **Jupyter Notebook:** Train, evaluate, and visualize model performance  
- 💾 **Modular Codebase:** Clean `src/` structure with reusable components  
- 🧩 **Sustainability Focus:** Supports smart waste segregation and eco-friendly solutions  

---

## 📂 Project Structure

sustainability_cnn/
├── data/ # dataset (not included)
├── models/ # trained models (.h5/.keras)
├── notebooks/
│ └── train_model.ipynb # training notebook
├── src/
│ ├── app_streamlit.py # Streamlit web app
│ ├── data_loader.py # data loading & preprocessing
│ ├── gradcam.py # Grad-CAM visualization
│ ├── model_builder.py # CNN / MobileNetV2 model
│ ├── predict.py # CLI prediction script
│ └── train.py # training script
├── requirements.txt
├── Dockerfile
├── Procfile
├── .gitignore
└── README.md



---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/sustainability_cnn.git
cd sustainability_cnn
2️⃣ Create and Activate Virtual Environment

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
3️⃣ Install Dependencies

pip install -r requirements.txt
4️⃣ Run the Web App

streamlit run src/app_streamlit.py
Visit http://localhost:8501 in your browser to use the app.

🧠 Model Details
Architecture: MobileNetV2 (pretrained on ImageNet)

Fine-tuning: Last 50 layers unfrozen for domain adaptation

Optimizer: Adam (LR: 1e-4 → 1e-5 during fine-tune)

Loss: Binary Crossentropy

Accuracy: ~92% on test set

### 🧠 Model Information
The model file used in this deployment is:
`models/waste_classifier_finetuned.h5` (~26 MB)

This is the **final fine-tuned MobileNetV2 model** trained for the highest accuracy (~92%).  
It is included in this repository for easy testing and deployment.


Explainability: Grad-CAM visualization for model transparency

📸 How It Works
User uploads a waste image (e.g., banana peel or plastic bottle).

The model processes the image (resized → normalized → prediction).

The output shows:

Predicted class: Organic / Recyclable

Confidence score (0–1)

Grad-CAM heatmap overlay to explain the decision

Example:

Input Image	                 Grad-CAM Overlay	                    Prediction
🍌 Banana Peel	        🔥 Focused on organic texture	            🌱 Organic (0.08)
🧴 Plastic Bottle	    🔥 Focused on bottle region	                 ♻️ Recyclable (0.93)

🌐 Deployment
🟢 Streamlit Community Cloud (Recommended)
Push your repo to GitHub.

Go to https://share.streamlit.io.

Click “New App” → select repo → path: src/app_streamlit.py.

Click Deploy — get a free public URL accessible from any device.

🐳 Docker Deployment

docker build -t waste-classifier .
docker run -p 8501:8501 waste-classifier
Open http://localhost:8501 to access your app.

📱 Mobile Access (via ngrok)

streamlit run src/app_streamlit.py
ngrok http 8501
Use the https:// ngrok URL on your phone.

🧩 Model File Management
The trained model (waste_classifier_finetuned.h5) is ignored by Git by default.

Use Git LFS or cloud storage (e.g., Google Drive, S3) for large files.

You can also convert to .keras or .tflite for lightweight deployment.

🧪 CLI Prediction
You can also classify a single image from the command line:


python src/predict.py --image path/to/image.jpg
💡 Future Improvements
🗑️ Expand to multi-class waste detection (Glass, Metal, Paper, E-waste)

📱 Export to TensorFlow Lite for mobile and IoT devices

🌈 Add side-by-side Grad-CAM comparison view

🚀 Integrate real-time camera feed for live classification

📈 Results Snapshot
Metric	Value
Train Accuracy	93.5%
Validation Accuracy	93.0%
Test Accuracy	91.8%
Loss	0.21
F1-Score (avg)	0.92

🧭 Purpose
This project supports the Sustainable Development Goals (SDG 12: Responsible Consumption and Production) by enabling automated and transparent waste segregation — a crucial step toward smarter, cleaner cities.

🧑‍💻 Author
Mayank Bhatt
import streamlit as st
import tempfile
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

# --------------------------
# 1. Device & Model Loading
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load EfficientNet-V2-S with a modified classifier
model = efficientnet_v2_s(weights=None)
in_features = model.classifier[1].in_features
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(in_features, 1024),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(1024),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(1024, 512),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(512),
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, 5)  # Ensure this matches your model's number of classes
)

# Load model weights
model_path = "best_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    raise FileNotFoundError(f"Model file '{model_path}' not found!")
model.to(device)
model.eval()

# --------------------------
# 2. Image Preprocessing
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found!")
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image, image_tensor

# --------------------------
# 3. Disease Information
# --------------------------
disease_info = {
    "Bacterial Pneumonia": "A lung infection caused by bacteria, leading to inflammation and fluid accumulation. Symptoms include fever, cough, chest pain, and difficulty breathing.",
    "Corona Virus Disease": "COVID-19 affects the respiratory system, causing symptoms like fever, dry cough, and breathing difficulties. Severe cases may lead to pneumonia and organ failure.",
    "Normal": "The lungs appear healthy with no signs of infection, inflammation, or disease. Maintain a healthy lifestyle to keep your lungs in good condition.",
    "Tuberculosis": "A bacterial infection that primarily affects the lungs. It spreads through airborne droplets and can cause persistent cough, weight loss, fever, and night sweats.",
    "Viral Pneumonia": "Caused by viruses like influenza or RSV, this type of pneumonia leads to inflammation in the lungs. Symptoms include dry cough, fever, fatigue, and shortness of breath."
}

# --------------------------
# 4. Severity Classification & Recommendations
# --------------------------
def classify_severity(percentage):
    if percentage < 10:
        return "Mild"
    elif 10 <= percentage < 30:
        return "Moderate"
    else:
        return "Severe"

def get_recommendation(pred_label, severity):
    detailed_recommendations = {
        "Bacterial Pneumonia": {
            "Mild": "Take prescribed antibiotics, rest, and drink plenty of fluids. Avoid exposure to smoke and allergens.",
            "Moderate": "Consult a doctor, take antibiotics as prescribed, monitor oxygen levels, and get chest physiotherapy if needed.",
            "Severe": "Hospitalization required. Oxygen therapy, IV antibiotics, and respiratory support may be necessary."
        },
        "Corona Virus Disease": {
            "Mild": "Self-isolate, take fever-reducing medications, stay hydrated, and monitor symptoms.",
            "Moderate": "Consult a doctor, get oxygen level monitoring, take prescribed antivirals, and maintain respiratory hygiene.",
            "Severe": "Hospitalization needed. Oxygen therapy, ventilatory support, and ICU care may be required."
        },
        "Normal": {
            "General": "No medical intervention required. Maintain a healthy lifestyle, exercise regularly, and avoid smoking."
        },
        "Tuberculosis": {
            "Mild": "Start anti-TB medication (DOTS therapy), ensure good nutrition, and avoid close contact with others.",
            "Moderate": "Strict adherence to medication, regular doctor visits, and avoid alcohol or smoking to prevent complications.",
            "Severe": "Hospitalization required for multi-drug resistant TB. Long-term treatment with strict isolation protocols."
        },
        "Viral Pneumonia": {
            "Mild": "Get adequate rest, stay hydrated, and use fever-reducing medications.",
            "Moderate": "Doctor consultation is necessary. Oxygen therapy and antivirals may be needed.",
            "Severe": "Hospitalization is required. Antiviral treatment, oxygen therapy, and ventilatory support may be necessary."
        }
    }
    
    return detailed_recommendations.get(pred_label, {}).get(severity, "Consult a healthcare professional.")


# --------------------------
# 3. Compute Red Area Percentage
# --------------------------
def calculate_red_area_percentage(heatmap):
    red_mask = heatmap[:, :, 2] > 150  # Threshold for strong red activation
    red_percentage = np.sum(red_mask) / (heatmap.shape[0] * heatmap.shape[1]) * 100
    return red_percentage

# --------------------------
# 5. Grad-CAM Explanation with Extra Visualizations
# --------------------------
# --------------------------
# 5. Grad-CAM Explanation with Extra Visualizations
# --------------------------
def explain_gradcam(image_path):
    image, image_tensor = preprocess_image(image_path)

    output = model(image_tensor)
    predicted_class = output.argmax(dim=1).item()
    predicted_label = classes[predicted_class]

    # Show disease description
    st.subheader("Disease Information")
    st.write(f"**{predicted_label}**: {disease_info.get(predicted_label, 'No information available.')}")

    # Grad-CAM heatmap computation
    gradients = None
    activations = None

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    target_layer = model.features[-1]
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    model.zero_grad()
    output[0, predicted_class].backward()

    if gradients is None or activations is None:
        handle_forward.remove()
        handle_backward.remove()
        st.error("Grad-CAM failed: No gradients or activations captured.")
        return

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = activations.squeeze(0)
    for i in range(activations.shape[0]):
        activations[i, :, :] *= pooled_gradients[i]

    heatmap = activations.mean(dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    red_percentage = calculate_red_area_percentage(heatmap_color)
    severity = classify_severity(red_percentage)
    recommendation = get_recommendation(predicted_label, severity)

    # Show images in Streamlit UI
    st.subheader("Grad-CAM Visualization")
    col1, col2 = st.columns(2)

    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

    with col2:
        st.image(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB), caption=f"Grad-CAM Heatmap\nPredicted: {predicted_label}\nRed Area: {red_percentage:.2f}%", use_column_width=True)

    # Display 3D surface plot
    X = np.arange(heatmap_resized.shape[1])
    Y = np.arange(heatmap_resized.shape[0])
    X, Y = np.meshgrid(X, Y)
    Z = heatmap_resized

    fig3d = plt.figure(figsize=(8, 6))
    ax3d = fig3d.add_subplot(111, projection='3d')
    surf = ax3d.plot_surface(X, Y, Z, cmap='jet', edgecolor='none')
    ax3d.set_title("3D Surface Plot of Grad-CAM Heatmap")
    fig3d.colorbar(surf, shrink=0.5, aspect=5)

    st.pyplot(fig3d)  # Show 3D plot in UI

    # Display text output
    st.subheader("Detailed Report")
    st.write(f"**Predicted Disease:** {predicted_label}")
    st.write(f"**Affected Area (%):** {red_percentage:.2f}%")
    st.write(f"**Severity Level:** {severity}")
    st.write(f"**Recommendation:** {recommendation}")

    handle_forward.remove()
    handle_backward.remove()


# --------------------------
# 6. Run Grad-CAM Explanation via Streamlit UI
# --------------------------
classes = ["Bacterial Pneumonia", "Corona Virus Disease", "Normal", "Tuberculosis", "Viral Pneumonia"]

st.title("Grad-CAM Visualization for Lung Disease Classification")
st.write("Upload a chest X-ray image and see the Grad-CAM results along with a detailed report.")

uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    
    st.image(Image.open(tmp_file_path), caption="Uploaded Image", use_column_width=True)
    
    st.write("Running Grad-CAM...")
    explain_gradcam(tmp_file_path)

import streamlit as st
import cv2
import numpy as np
from skimage import color
from tensorflow.keras.models import load_model
import tempfile
import os

@st.cache_resource
def load_colorization_model():
    try:
        # Replace with your actual model path
        model = load_model(r'C:\Users\Asus\Saved Models\generator_model.keras')  
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    

def preprocess_image(img, target_size=(256, 256)):
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = img.shape[:2]
    scale = min(target_size[0]/h, target_size[1]/w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    delta_h = target_size[0] - new_h
    delta_w = target_size[1] - new_w
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, 
                                   cv2.BORDER_REFLECT)
    
    img_lab = color.rgb2lab(img_padded)
    
    img_lab = img_lab.astype('float32')
    img_lab[..., 0] = img_lab[..., 0] / 100.0  
    img_lab[..., 1:] = (img_lab[..., 1:] + 128) / 255.0
    
    return img_lab, img_padded

def main():
    st.title("🎨 Neural Image Colorization")
    st.markdown("Upload a grayscale image and let our AI model add color to it!")
    
    # Load model
    generator = load_colorization_model()
    if generator is None:
        st.stop()
    
    # File upload
    uploaded_file = st.file_uploader("Choose an image...", 
                                    type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Save uploaded file to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # Read image
        img = cv2.imread(tmp_path)
        os.unlink(tmp_path)  # Delete temp file
        
        if img is None:
            st.error("Error: Could not read image. Please try another file.")
            return
        
        # Preprocess image
        with st.spinner('Processing image...'):
            lab_img, img_padded = preprocess_image(img)
            L_channel = lab_img[..., 0]
        
        # Display original and L channel
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                     use_column_width=True)
        
        with col2:
            st.subheader("Grayscale Input")
            st.image(L_channel, 
                     caption="L Channel (model input)", 
                     clamp=True, 
                     use_column_width=True)
        
        # Predict colorization
        with st.spinner('Colorizing image...'):
            input_L = np.expand_dims(L_channel, axis=(0, -1))  # Add batch and channel dims
            ab_pred = generator.predict(input_L)[0]
            
            # Reconstruct LAB image
            L_denorm = L_channel * 100.0
            ab_denorm = ab_pred * 255.0 - 128.0
            pred_lab = np.stack([L_denorm, ab_denorm[..., 0], ab_denorm[..., 1]], axis=-1)
            colorized_rgb = color.lab2rgb(pred_lab)
        
        # Display results
        st.subheader("Colorized Result")
        st.image(colorized_rgb, 
                 caption="AI Colorized Image", 
                 use_column_width=True)
        
        # Download button
        result_img = (colorized_rgb * 255).astype(np.uint8)
        _, buf = cv2.imencode(".png", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        st.download_button(
            label="Download Colorized Image",
            data=buf.tobytes(),
            file_name="colorized_image.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()

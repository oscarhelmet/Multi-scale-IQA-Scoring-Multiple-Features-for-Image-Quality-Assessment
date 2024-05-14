from flask import Flask, render_template, request, jsonify, flash, redirect
from FR.model import Multiscale, extract_features
import torch
from NR.model import MetaIQA 
import cv2
from torchvision import transforms
import numpy as np
import pillow_heif

pillow_heif.register_heif_opener()

app = Flask(__name__)


try:
    model = Multiscale()
    model.load_model('FR/model_multi_0.001_128_800.pt')
    model.eval()

    options = {'gpu': True}  # Adjust according to server capabilities
    model2 = MetaIQA(options)
    model2.load_model('NR/metaiqa.pth')
    model2.eval()

except Exception as e:
    app.logger.error(f"Failed to load model: {e}")



def load_image_from_file(file):
    """Load an image from an uploaded file."""

    in_memory_file = file.read()
    image = cv2.imdecode(np.frombuffer(in_memory_file, np.uint8), cv2.IMREAD_COLOR)
    print(image.shape)
    if image is None:
        raise ValueError("Unable to decode the image file.")
    return image

def preprocess_image(image, size=(224, 224)):
    """Preprocess the image to feed into the neural network."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def heic_to_jpeg(file):
    try:
        heif_file = pillow_heif.open_heif(file)
        print(np.asarray(heif_file).shape)
        # image = cv2.imdecode(np.frombuffer(heif_file.data, np.uint8), cv2.IMREAD_COLOR)
        
        return preprocess_image(np.asarray(heif_file))
       
    except Exception as e:
        raise ValueError(f"Failed to convert HEIC to JPEG: {e}")

@app.route('/NR', methods=['GET', 'POST'])
def nr_iqa():
    if request.method == 'POST':
        if 'target_image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        target_file = request.files['target_image']
        if target_file.filename == '':
            flash('No selected file')
            return redirect(request.url)
       
        

        try:
            if target_file.filename.endswith('.heic'):
                image = heic_to_jpeg(target_file)
            else:
                image = load_image_from_file(target_file)
                image = preprocess_image(image)

            # Initialize and load the model

            image = image.unsqueeze(0)  # Add batch dimension

            # Check for GPU availability
            if torch.cuda.is_available():
                image = image.cuda()

            # Forward pass through the model
            with torch.no_grad():
                quality_score = model2(image)
                quality_score = quality_score/3.15*10**1.74

            return render_template('nr_iqa.html', score="{:.2f}".format(quality_score.item()))
        except Exception as e:
            app.logger.error(f"Error processing image: {e}")
            return jsonify({'error': 'Internal Server Error'}), 500

    return render_template('nr_iqa.html', score=None)



@app.route('/FR', methods=['GET', 'POST'])
def fr_iqa():
    if request.method == 'POST':
        if 'target_image' not in request.files or 'reference_image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        target_file = request.files['target_image']
        reference_file = request.files['reference_image']
        if target_file.filename == '' or reference_file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if target_file and reference_file:
            # Process files
            layer_names = ['conv1_1','conv2_2','conv3_3','conv4_3','conv5_3']
            assessed_features = extract_features(target_file, layer_names)
            reference_features = extract_features(reference_file, layer_names)
            
            diff = assessed_features - reference_features
            diff_tensor = torch.tensor(diff, dtype=torch.float32).unsqueeze(0)
            output = model(diff_tensor)
            prediction = torch.argmax(output, dim=1)

            return render_template('fr_iqa.html', score=prediction.item())
    return render_template('fr_iqa.html', score=None)


@app.route('/')
def index():
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True, port=8000)
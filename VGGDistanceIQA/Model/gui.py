import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from model import Multiscale, extract_features
import numpy as np

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack(fill=tk.BOTH, expand=1)
        self.create_widgets()
        

    def create_widgets(self):
        # Button to load the model
        self.load_model_button = tk.Button(self, text="Load Model", command=self.load_model)
        self.load_model_button.pack(side="top", pady=10)

        # Button to load the assessed image
        self.load_assessed_image_button = tk.Button(self, text="Load Assessed Image", command=lambda: self.load_image("assessed"))
        self.load_assessed_image_button.pack(side="top", pady=10)

        # Button to load the reference image
        self.load_reference_image_button = tk.Button(self, text="Load Reference Image", command=lambda: self.load_image("reference"))
        self.load_reference_image_button.pack(side="top", pady=10)

        # Button to rate the photo
        self.rate_button = tk.Button(self, text="Rate", command=self.rate_photo)
        self.rate_button.pack(side="top", pady=10)

        # Quit button
        self.quit_button = tk.Button(self, text="QUIT", fg="red", command=self.master.destroy)
        self.quit_button.pack(side="bottom", pady=10)

    def load_model(self):
        self.model_path = filedialog.askopenfilename(filetypes=[("PyTorch Models", "*.pt *.pth")])
        self.model = Multiscale()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        print("Model loaded.")

    def load_image(self, image_type):
        image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpeg *.jpg *.png *.bmp *.gif")])
        if image_type == "assessed":
            self.assessed_image_path = image_path
        elif image_type == "reference":
            self.reference_image_path = image_path
        
        # Display the image
        pil_image = Image.open(image_path)
        pil_image = pil_image.resize((640, 360), Image.Resampling.LANCZOS)
        tk_image = ImageTk.PhotoImage(pil_image)
        if hasattr(self, 'image_label'):
            self.image_label.configure(image=tk_image)
        else:
            self.image_label = tk.Label(image=tk_image)
            self.image_label.pack(side="top", pady=10)
        self.image_label.image = tk_image  # keep a reference!

    def rate_photo(self):
        

        layer_names = ['conv1_1','conv2_2','conv3_3','conv4_3','conv5_3']
        features_assessed = extract_features(self.assessed_image_path, layer_names)
        features_reference = extract_features(self.reference_image_path, layer_names)
        
        diff = features_reference - features_assessed
        diff_tensor = torch.tensor(diff, dtype=torch.float32).unsqueeze(0)

        output = self.model(diff_tensor)
        prediction = torch.argmax(output, dim=1)
        result_text = f"score: {prediction.item()}"
        if hasattr(self, 'result_label'):
            self.result_label.configure(text=result_text)
        else:
            self.result_label = tk.Label(text=result_text)
            self.result_label.pack(side="top", pady=10)



root = tk.Tk()
root.geometry("1280x720")
app = Application(master=root)
app.mainloop()
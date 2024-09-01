import tkinter as tk
import customtkinter
from tkinter import *
from tkinter import filedialog,messagebox
from PIL import Image, ImageTk

import os
import cv2
import time
import imutils
import pickle
import psutil
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

#######################################
# env yg harus diganti itu model path
#######################################

# Load the pre-trained kNN and SVM models
data_path = r"C:\\Users\SMSK_Narinda\Desktop\Research\dataset"
knn_model_path = r'C:\Users\SMSK_Narinda\Desktop\Skripsi_Checkpoint_KNN_SVM_Dermnet\Model\KNNClassifier.p'
svm_model_path = r'C:\Users\SMSK_Narinda\Desktop\Skripsi_Checkpoint_KNN_SVM_Dermnet\Model\SVM_71_Classifier.p'

with open(knn_model_path, 'rb') as file:
    knn = pickle.load(file)

with open(svm_model_path, 'rb') as file:
    svm = pickle.load(file)

true_labels = []
for label in os.listdir(data_path):
    true_labels.append(label)

# Function to resize image while maintaining aspect ratio
def resize_image(image, max_size):
    ratio = min(max_size / image.width, max_size / image.height)
    new_size = (int(image.width * ratio), int(image.height * ratio))
    return image.resize(new_size, Image.Resampling.LANCZOS)

# Create extract color from image function.  
def extract_color_histogram(image, bins=(32,32,32)):
    # Extract 3d color by its hue, saturation, and vue (HSV)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, bins,
                       [0, 180, 0 , 256, 0, 256])
    
    ### handle normalizing the historigram
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else :
        cv2.normalize(hist,hist)
        
    return hist.flatten()

# Preprocessing SVM
def transform_labels(labels):
    label_mapping = {label: idx for idx, label in enumerate(np.unique(labels))}
    return np.array([label_mapping[label] for label in labels]), label_mapping

def inverse_transform_labels(predictions, label_mapping):
    inverse_mapping = {v: k for k, v in label_mapping.items()}
    transformed_labels = []
    for pred in predictions:
        rounded_pred = int(round(pred))
        if rounded_pred in inverse_mapping:
            transformed_labels.append(inverse_mapping[rounded_pred])
        else:
            print(f"Unexpected prediction value: {rounded_pred}")
            transformed_labels.append("Unknown")  # Or handle it appropriately
    return np.array(transformed_labels)


# GUI Application
class ImageClassifierApp:
    def __init__(self, true_labels, data_path):
        super().__init__()
        self.true_labels = true_labels
        self.data_path = data_path
        self.label_match = None  # Initialize accuracy attribute
        self.test_img = None
        self.accuracy = None
        self.root = root
        self.root.title("Image Classifier")
        self.root.geometry('560x1080')  # Set the window size to 500x500
        
        self.header = customtkinter.CTkLabel(root, text="Prediksi Gambar Menggunakan Model Klasifikasi Machine Learning",
                                              font=("Arial", 16))
        self.header.pack(pady=20)

        frame1 = tk.Frame(master=root, width=100, height=100, bg='gray16')
        frame1.pack(fill=tk.X)
        self.label = customtkinter.CTkLabel(frame1, text="Cari Gambar:")
        self.label.pack(pady=1)
        self.upload_button = customtkinter.CTkButton(frame1, text="Buka File", command=self.upload_image)
        self.upload_button.pack(pady=3)

        frame2 = tk.Frame(master=root, width=100, height=100, bg='gray16')
        frame2.pack(fill=tk.X)
        self.label = customtkinter.CTkLabel(frame2, text="Pilih Model Klasifikasi:")
        self.label.pack(pady=1)
        self.model_var = tk.StringVar(value='knn')
        self.knn_radio = customtkinter.CTkRadioButton(frame2, text="KNN", variable=self.model_var, value='knn')
        self.knn_radio.pack(pady=3)
        self.svm_radio = customtkinter.CTkRadioButton(frame2, text="SVM", variable=self.model_var, value='svm')
        self.svm_radio.pack(pady=3)


        self.classify_button = customtkinter.CTkButton(root, text="Classify", command=self.classify_image)
        self.classify_button.pack(pady=3)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.result_memory = customtkinter.CTkLabel(root, text="", font=("Arial", 12))
        self.result_memory.pack(pady=10)

        self.result_time = customtkinter.CTkLabel(root, text="", font=("Arial", 12))
        self.result_time.pack(pady=10)

        self.result_label = customtkinter.CTkLabel(root, text="", font=("Arial", 12))
        self.result_label.pack(pady=10)

        self.accuracy_label = customtkinter.CTkLabel(root, text="", font=("Arial", 12))
        self.accuracy_label.pack(pady=10)


    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*"))
        )
        if file_path:
            self.image_path = file_path
            self.original_image = Image.open(file_path)
            img_resized = resize_image(self.original_image, 500)
            img_resized_tk = ImageTk.PhotoImage(img_resized)
            self.image_label.configure(image=img_resized_tk)
            self.image_label.image = img_resized_tk
            self.test_img = cv2.imread(file_path)

            self.img_name = os.path.splitext(os.path.basename(file_path))[0]


    def classify_image(self):
        if hasattr(self, 'test_img') is not None:
            img = self.test_img
            true_labels = [label for label in os.listdir(self.data_path)]

            # Define current memory and time
            current_process = psutil.Process(os.getpid())
            mem_start = current_process.memory_info().rss
            start_time = time.time()
            
            selected_model = self.model_var.get()
            if selected_model == 'knn':
                print("Working on KNN Classifier...")
                # Preprocessing Image
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                test_image = extract_color_histogram(img)
                test_feature = np.array(test_image).reshape(1, -1)

                # Result Testing
                self.prediction = knn.predict(test_feature)[0]
                print(self.prediction)


            elif selected_model == 'svm': 
                print("Working on SVM Classifier...")

                labels_train_transformed, label_mapping = transform_labels(true_labels)
                true_labels = [label for label in os.listdir(self.data_path)]


                # Preprocessing Image
                img = cv2.resize(img, (150,150))
                img = img.flatten()
                img = np.array(img)
                predicted_class = svm.predict([img])
                print("predicted class:", predicted_class)
                prediction_list = inverse_transform_labels(predicted_class, label_mapping)
                self.prediction = prediction_list[0]
                print("prediction :", self.prediction)
            
            # Accuracy label comparation
            print(self.img_name)

            prediction_name_words = []
            for char in self.prediction:
                if char in ('_', ' ', '-'):  # Check for both underscore and space
                    # If the character is a separator, add a new word
                    prediction_name_words.append('')
                else:
                    # If the character is not a separator, append it to the current word
                    if prediction_name_words:  # Check if any words have been added yet
                        prediction_name_words[-1] += char
                    else:
                        prediction_name_words.append(char)
            print(prediction_name_words)

            image_name_words = []
            for char in self.img_name:
                if char in ('_', ' ', '-'):  # Check for both underscore and space
                    # If the character is a separator, add a new word
                    image_name_words.append('')
                else:
                    # If the character is not a separator, append it to the current word
                    if image_name_words:  # Check if any words have been added yet
                        image_name_words[-1] += char
                    else:
                        image_name_words.append(char)
            print(image_name_words) 

            def any_keyword_present(image_name_words, prediction_words):
                # Ubah kedua list menjadi huruf kecil agar cocok tanpa memperhatikan huruf besar/kecil
                image_name_words = [word.lower() for word in image_name_words]
                prediction_words = [word.lower() for word in prediction_words]

                # Periksa apakah ada kata dalam prediction_words yang ada dalam image_name_words
                return any(word in image_name_words for word in prediction_words)
            
            is_present = any_keyword_present(image_name_words, prediction_name_words)
            accuracy = float(is_present)

            # Measure memory and time for prediction
            end_time = time.time()
            mem_end = current_process.memory_info().rss
            memory_usage = ((mem_end - mem_start)) / 1024  # Memory usage in KB
            time_taken = (end_time - start_time) * 1000  # Time taken in milliseconds


            self.result_label.configure(text=f'Predicted Digit: {self.prediction}')
            self.result_memory.configure(text=f'Memory Usage: {memory_usage: .2f} kB')
            self.result_time.configure(text=f'Duration: {time_taken:.2f} ms')
            self.accuracy_label.configure(text=f'Accuracy: {accuracy:.2f}')

        else:
            messagebox.showerror("Error", "Please upload an image first.")
        
    

if __name__ == "__main__":
    data_path = r"C:\\Users\SMSK_Narinda\Desktop\Research\dataset"
    true_labels = []
    for label in os.listdir(data_path):
        true_labels.append(label)
    root = customtkinter.CTk()
    app = ImageClassifierApp(true_labels, data_path)
    root.mainloop()

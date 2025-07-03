import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
from ultralytics import YOLO
import os
import threading
import time

class AnimalDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Animal Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Animal classes from your dataset
        self.class_names = {
            0: "Dog", 1: "Cat", 2: "Zebra", 3: "Lion", 4: "Leopard",
            5: "Cheetah", 6: "Tiger", 7: "Bear", 8: "Brown Bear", 9: "Butterfly",
            10: "Canary", 11: "Crocodile", 12: "Polar Bear", 13: "Bull", 14: "Camel",
            15: "Crab", 16: "Chicken", 17: "Centipede", 18: "Cattle", 19: "Caterpillar", 20: "Duck"
        }
        
        # Carnivorous animals (highlighted in red)
        self.carnivorous_animals = {
            "Lion", "Leopard", "Cheetah", "Tiger", "Bear", "Brown Bear", 
            "Polar Bear", "Crocodile", "Cat"
        }
        
        # Initialize variables
        self.model = None
        self.current_image = None
        self.current_video_path = None
        self.video_cap = None
        self.is_playing = False
        self.detection_thread = None
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        # Main title
        title_label = tk.Label(self.root, text="Animal Detection System", 
                              font=("Arial", 20, "bold"), bg='#f0f0f0', fg='#333')
        title_label.pack(pady=10)
        
        # Control frame
        control_frame = tk.Frame(self.root, bg='#f0f0f0')
        control_frame.pack(pady=10)
        
        # Buttons
        self.load_model_btn = tk.Button(control_frame, text="Load Model", 
                                       command=self.load_model, font=("Arial", 12),
                                       bg='#4CAF50', fg='white', padx=20)
        self.load_model_btn.grid(row=0, column=0, padx=5)
        
        self.load_image_btn = tk.Button(control_frame, text="Load Image", 
                                       command=self.load_image, font=("Arial", 12),
                                       bg='#2196F3', fg='white', padx=20)
        self.load_image_btn.grid(row=0, column=1, padx=5)
        
        self.load_video_btn = tk.Button(control_frame, text="Load Video", 
                                       command=self.load_video, font=("Arial", 12),
                                       bg='#FF9800', fg='white', padx=20)
        self.load_video_btn.grid(row=0, column=2, padx=5)
        
        self.detect_btn = tk.Button(control_frame, text="Detect Animals", 
                                   command=self.detect_animals, font=("Arial", 12),
                                   bg='#9C27B0', fg='white', padx=20)
        self.detect_btn.grid(row=0, column=3, padx=5)
        
        # Video controls
        video_control_frame = tk.Frame(self.root, bg='#f0f0f0')
        video_control_frame.pack(pady=5)
        
        self.play_btn = tk.Button(video_control_frame, text="Play Video", 
                                 command=self.play_video, font=("Arial", 10),
                                 bg='#4CAF50', fg='white', state='disabled')
        self.play_btn.grid(row=0, column=0, padx=5)
        
        self.pause_btn = tk.Button(video_control_frame, text="Pause Video", 
                                  command=self.pause_video, font=("Arial", 10),
                                  bg='#F44336', fg='white', state='disabled')
        self.pause_btn.grid(row=0, column=1, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(pady=5, fill=tk.X, padx=50)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Ready to load model...", 
                                    font=("Arial", 10), bg='#f0f0f0', fg='#666')
        self.status_label.pack(pady=5)
        
        # Display frame
        display_frame = tk.Frame(self.root, bg='#f0f0f0')
        display_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        
        # Original and detected image frames
        self.original_frame = tk.Frame(display_frame, bg='white', relief=tk.RAISED, bd=2)
        self.original_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 10))
        
        self.detected_frame = tk.Frame(display_frame, bg='white', relief=tk.RAISED, bd=2)
        self.detected_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=(10, 0))
        
        # Labels for images
        tk.Label(self.original_frame, text="Original", font=("Arial", 14, "bold")).pack(pady=5)
        tk.Label(self.detected_frame, text="Detected", font=("Arial", 14, "bold")).pack(pady=5)
        
        # Canvas for images
        self.original_canvas = tk.Canvas(self.original_frame, bg='white')
        self.original_canvas.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        self.detected_canvas = tk.Canvas(self.detected_frame, bg='white')
        self.detected_canvas.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
    
    def load_model(self):
        """Load the YOLO model"""
        try:
            self.status_label.config(text="Loading model...")
            self.progress.start()
            
            # You can change this path to your trained model
            model_path = filedialog.askopenfilename(
                title="Select YOLO Model",
                filetypes=[("Model files", "*.pt"), ("All files", "*.*")]
            )
            
            if model_path:
                self.model = YOLO(model_path)
                self.status_label.config(text=f"Model loaded successfully: {os.path.basename(model_path)}")
                messagebox.showinfo("Success", "Model loaded successfully!")
            else:
                # If no model selected, try to use YOLOv8 pre-trained model
                self.model = YOLO('yolov8n.pt')  # You can change this to your custom model
                self.status_label.config(text="Using default YOLOv8 model")
                
            self.progress.stop()
            
        except Exception as e:
            self.progress.stop()
            self.status_label.config(text="Error loading model")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def load_image(self):
        """Load an image for detection"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.current_image = cv2.imread(file_path)
                self.display_original_image(self.current_image)
                self.status_label.config(text=f"Image loaded: {os.path.basename(file_path)}")
                
                # Reset video-related variables
                self.current_video_path = None
                self.is_playing = False
                self.play_btn.config(state='disabled')
                self.pause_btn.config(state='disabled')
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def load_video(self):
        """Load a video for detection"""
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.current_video_path = file_path
                self.video_cap = cv2.VideoCapture(file_path)
                
                # Read first frame to display
                ret, frame = self.video_cap.read()
                if ret:
                    self.current_image = frame
                    self.display_original_image(frame)
                    self.status_label.config(text=f"Video loaded: {os.path.basename(file_path)}")
                    
                    # Enable video controls
                    self.play_btn.config(state='normal')
                    self.pause_btn.config(state='normal')
                
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load video: {str(e)}")
    
    def display_original_image(self, image):
        """Display original image on canvas"""
        try:
            # Resize image to fit canvas
            height, width = image.shape[:2]
            canvas_width = self.original_canvas.winfo_width()
            canvas_height = self.original_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # Calculate scaling factor
                scale = min(canvas_width/width, canvas_height/height, 1.0)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                # Resize image
                resized_image = cv2.resize(image, (new_width, new_height))
                
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Clear canvas and display image
                self.original_canvas.delete("all")
                self.original_canvas.create_image(canvas_width//2, canvas_height//2, 
                                                image=photo, anchor=tk.CENTER)
                self.original_canvas.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Error displaying original image: {str(e)}")
    
    def display_detected_image(self, image):
        """Display detected image on canvas"""
        try:
            # Resize image to fit canvas
            height, width = image.shape[:2]
            canvas_width = self.detected_canvas.winfo_width()
            canvas_height = self.detected_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # Calculate scaling factor
                scale = min(canvas_width/width, canvas_height/height, 1.0)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                # Resize image
                resized_image = cv2.resize(image, (new_width, new_height))
                
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Clear canvas and display image
                self.detected_canvas.delete("all")
                self.detected_canvas.create_image(canvas_width//2, canvas_height//2, 
                                                image=photo, anchor=tk.CENTER)
                self.detected_canvas.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Error displaying detected image: {str(e)}")
    
    def detect_animals(self):
        """Perform animal detection on current image"""
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first!")
            return
        
        if self.current_image is None:
            messagebox.showerror("Error", "Please load an image or video first!")
            return
        
        try:
            self.status_label.config(text="Detecting animals...")
            self.progress.start()
            
            # Perform detection
            results = self.model(self.current_image)
            
            # Process results
            detected_image = self.current_image.copy()
            carnivorous_count = 0
            detected_animals = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.class_names.get(class_id, f"Class_{class_id}")
                        detected_animals.append(class_name)
                        
                        # Check if carnivorous
                        is_carnivorous = class_name in self.carnivorous_animals
                        if is_carnivorous:
                            carnivorous_count += 1
                        
                        # Choose color (red for carnivorous, green for others)
                        color = (0, 0, 255) if is_carnivorous else (0, 255, 0)
                        
                        # Draw bounding box
                        cv2.rectangle(detected_image, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{class_name}: {confidence:.2f}"
                        if is_carnivorous:
                            label += " (CARNIVOROUS)"
                        
                        # Calculate text size and draw background
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        
                        cv2.rectangle(detected_image, (x1, y1 - text_height - baseline - 5),
                                    (x1 + text_width, y1), color, -1)
                        
                        # Draw text
                        cv2.putText(detected_image, label, (x1, y1 - baseline - 2),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display detected image
            self.display_detected_image(detected_image)
            
            # Show carnivorous count popup
            if carnivorous_count > 0:
                messagebox.showinfo("Carnivorous Animals Detected!", 
                                  f"Number of carnivorous animals detected: {carnivorous_count}\n\n" +
                                  f"Detected animals: {', '.join(set(detected_animals))}")
            else:
                messagebox.showinfo("Detection Complete", 
                                  f"No carnivorous animals detected.\n\n" +
                                  f"Detected animals: {', '.join(set(detected_animals)) if detected_animals else 'None'}")
            
            self.status_label.config(text="Detection completed!")
            self.progress.stop()
            
        except Exception as e:
            self.progress.stop()
            self.status_label.config(text="Error during detection")
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
    
    def play_video(self):
        """Play video with real-time detection"""
        if self.current_video_path is None:
            messagebox.showerror("Error", "Please load a video first!")
            return
        
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first!")
            return
        
        self.is_playing = True
        self.play_btn.config(state='disabled')
        self.pause_btn.config(state='normal')
        
        # Start video processing in a separate thread
        self.detection_thread = threading.Thread(target=self.process_video)
        self.detection_thread.daemon = True
        self.detection_thread.start()
    
    def pause_video(self):
        """Pause video playback"""
        self.is_playing = False
        self.play_btn.config(state='normal')
        self.pause_btn.config(state='disabled')
    
    def process_video(self):
        """Process video frames with detection"""
        try:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            
            while self.is_playing:
                ret, frame = self.video_cap.read()
                if not ret:
                    break
                
                # Perform detection on frame
                results = self.model(frame)
                detected_frame = frame.copy()
                carnivorous_count = 0
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            
                            class_name = self.class_names.get(class_id, f"Class_{class_id}")
                            is_carnivorous = class_name in self.carnivorous_animals
                            
                            if is_carnivorous:
                                carnivorous_count += 1
                            
                            color = (0, 0, 255) if is_carnivorous else (0, 255, 0)
                            
                            cv2.rectangle(detected_frame, (x1, y1), (x2, y2), color, 2)
                            
                            label = f"{class_name}: {confidence:.2f}"
                            if is_carnivorous:
                                label += " (CARN)"
                            
                            cv2.putText(detected_frame, label, (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Update display
                self.root.after(0, lambda f=frame: self.display_original_image(f))
                self.root.after(0, lambda f=detected_frame: self.display_detected_image(f))
                
                # Add carnivorous count to frame
                if carnivorous_count > 0:
                    cv2.putText(detected_frame, f"Carnivorous: {carnivorous_count}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Control frame rate
                time.sleep(0.03)  # ~30 FPS
            
            # Reset video controls
            self.root.after(0, lambda: self.play_btn.config(state='normal'))
            self.root.after(0, lambda: self.pause_btn.config(state='disabled'))
            
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            self.is_playing = False

def main():
    root = tk.Tk()
    app = AnimalDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
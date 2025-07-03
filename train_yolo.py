"""
CPU-Optimized YOLOv8 Animal Detection Training Script
Specifically designed for systems without CUDA GPU support
"""

from ultralytics import YOLO # type: ignore
import os
import yaml # type: ignore
import torch # type: ignore
from pathlib import Path
import multiprocessing

def train_yolo_model_cpu_optimized():
    """Train YOLOv8 model optimized for CPU"""
    
    # CPU-Optimized Configuration
    DATA_CONFIG = r"C:\Users\Lutifah\Desktop\INTERSHIP\Animal_Detection_YOLOV8-main\animal_data.yaml"
    MODEL_SIZE = "yolov8n.pt"  # Use nano model for faster CPU training
    EPOCHS = 5  # Reduced epochs for CPU
    IMAGE_SIZE = 416  # Smaller image size for faster processing
    BATCH_SIZE = 2  # Small batch size for CPU
    
    # CPU optimization settings
    num_workers = min(4, multiprocessing.cpu_count())  # Limit workers
    
    # Project and run names
    PROJECT_NAME = "animal_detection_cpu"
    RUN_NAME = "yolov8_animals_cpu"
    
    try:
        # Check system capabilities
        print("="*60)
        print("SYSTEM INFORMATION")
        print("="*60)
        print(f"CPU Cores: {multiprocessing.cpu_count()}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Training Device: CPU")
        print("="*60)
        
        # Load the model
        print(f"\\nLoading YOLOv8 model: {MODEL_SIZE}")
        model = YOLO(MODEL_SIZE)
        
        # Verify data configuration file exists
        if not os.path.exists(DATA_CONFIG):
            print(f"Error: Data configuration file not found at {DATA_CONFIG}")
            print("Please make sure the data.yaml file is in the correct location.")
            return
        
        # Display training configuration
        print("\\n" + "="*60)
        print("CPU-OPTIMIZED TRAINING CONFIGURATION")
        print("="*60)
        print(f"Model: {MODEL_SIZE} (Nano - fastest)")
        print(f"Data Config: {DATA_CONFIG}")
        print(f"Epochs: {EPOCHS} (reduced for CPU)")
        print(f"Image Size: {IMAGE_SIZE} (optimized for CPU)")
        print(f"Batch Size: {BATCH_SIZE} (CPU optimized)")
        print(f"Workers: {num_workers}")
        print(f"Device: CPU")
        print(f"Project: {PROJECT_NAME}")
        print(f"Run Name: {RUN_NAME}")
        print("="*60)
        
        # CPU Training Tips
        print("\\n" + "="*60)
        print("CPU TRAINING TIPS")
        print("="*60)
        print("• Training on CPU will be slower than GPU")
        print("• Consider using a smaller dataset for testing")
        print("• Lower image resolution speeds up training")
        print("• Smaller batch sizes prevent memory issues")
        print("• Be patient - CPU training takes longer!")
        print("="*60)
        
        # Ask for confirmation
        proceed = input("\\nProceed with CPU training? This may take several hours (y/n): ").lower().strip()
        if proceed not in ['y', 'yes']:
            print("Training cancelled.")
            return
        
        # Start training with CPU optimizations
        print("\\nStarting CPU-optimized training...")
        print("Note: This will take significantly longer than GPU training.")
        
        results = model.train(
            data=DATA_CONFIG,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            project=PROJECT_NAME,
            name=RUN_NAME,
            save=True,
            plots=True,
            verbose=True,
            patience=15,  # Increased patience for CPU training
            save_period=5,  # Save more frequently
            workers=num_workers,
            device='cpu',
            
            # CPU optimization parameters
            amp=False,  # Disable automatic mixed precision (GPU feature)
            profile=False,  # Disable profiling to save resources
            
            # Additional CPU optimizations
            lr0=0.001,  # Lower learning rate for stability
            warmup_epochs=3,  # Reduced warmup
            close_mosaic=5,  # Disable mosaic augmentation in last epochs
        )
        
        print("\\nTraining completed successfully!")
        print(f"Best model saved at: {results.save_dir}/weights/best.pt")
        print(f"Last model saved at: {results.save_dir}/weights/last.pt")
        
        # Validate the model
        print("\\nValidating model...")
        metrics = model.val()
        print(f"Validation mAP50: {metrics.box.map50:.4f}")
        print(f"Validation mAP50-95: {metrics.box.map:.4f}")
        
        # Performance tips
        print("\\n" + "="*60)
        print("POST-TRAINING NOTES")
        print("="*60)
        print("• Your model is now ready for inference")
        print("• CPU inference will also be slower than GPU")
        print("• Consider using the model on smaller images for faster detection")
        print("• The nano model balances speed and accuracy for CPU use")
        print("="*60)
        
        return results
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None

def quick_cpu_test():
    """Quick test to verify CPU training setup"""
    
    print("Running quick CPU compatibility test...")
    
    try:
        # Test YOLO loading
        model = YOLO("yolov8n.pt")
        print(" YOLOv8 model loaded successfully")
        
        # Test CPU device
        device = 'cpu'
        print(f" Using device: {device}")
        
        # Test data config
        data_config = r"C:\\Users\\Lutifah\\Desktop\\INTERSHIP\\Animal_Detection_YOLOV8-main\\animal_data.yaml"
        if os.path.exists(data_config):
            print(" Data configuration file found")
        else:
            print("Data configuration file not found")
            print(f"Expected location: {data_config}")
        
        # Test system resources
        cores = multiprocessing.cpu_count()
        print(f" CPU cores available: {cores}")
        
        print("\\n" + "="*50)
        print("SYSTEM READY FOR CPU TRAINING!")
        print("="*50)
        print("Estimated training time: 2-6 hours (depending on dataset size)")
        print("Recommendation: Start training when you can leave it running")
        
    except Exception as e:
        print(f" Error in compatibility test: {str(e)}")

def main():
    """Main function for CPU-optimized training"""
    
    print("YOLOv8 Animal Detection - CPU Training")
    print("="*50)
    
    print("1. Run compatibility test")
    print("2. Start CPU-optimized training") 
    print("3. Exit")
    
    choice = input("\\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        quick_cpu_test()
    elif choice == '2':
        train_yolo_model_cpu_optimized()
    elif choice == '3':
        print("Goodbye!")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
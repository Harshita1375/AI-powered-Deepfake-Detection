import os
import zipfile
import cv2
import shutil
from pathlib import Path

# --- CONFIGURATION ---
downloads_path = str(os.path.join(Path.home(), "Downloads"))
ZIP_NAME = "archive (5).zip"  
ZIP_PATH = os.path.join(downloads_path, ZIP_NAME)

TEMP_DIR = "temp_extract"
IMAGE_ROOT = "data/image/raw"
TARGET_COUNT = 2833 

def count_existing_images(label):
    path = os.path.join(IMAGE_ROOT, label)
    if not os.path.exists(path):
        return 0
    return len([f for f in os.listdir(path) if f.endswith('.jpg')])

def run_extraction(interval=30):
    print(f"🔍 Searching for ZIP at: {ZIP_PATH}")
    if not os.path.exists(ZIP_PATH):
        print(f"❌ Error: ZIP not found.")
        return

    os.makedirs(IMAGE_ROOT, exist_ok=True)

    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            all_files = [f for f in z.namelist() if f.lower().endswith('.mp4')]
            
            # Initial counts
            fake_count = count_existing_images('fake')
            real_count = count_existing_images('real')

            print(f"📊 Current Progress: Fake: {fake_count}/{TARGET_COUNT} | Real: {real_count}/{TARGET_COUNT}")

            for file_path in all_files:
                # Check counts every loop
                fake_count = count_existing_images('fake')
                real_count = count_existing_images('real')

                label = 'real' if 'original' in file_path.lower() else 'fake'
                
                # --- SWITCH LOGIC ---
                # Skip if we already have enough for this specific category
                if label == 'fake' and fake_count >= TARGET_COUNT:
                    continue
                if label == 'real' and real_count >= TARGET_COUNT:
                    continue

                video_basename = os.path.basename(file_path).replace(".mp4", "")
                dest_folder = os.path.join(IMAGE_ROOT, label)
                os.makedirs(dest_folder, exist_ok=True)

                # Skip if specific video already processed
                if os.path.exists(os.path.join(dest_folder, f"{video_basename}_f0.jpg")):
                    continue 

                # 1. Extract
                z.extract(file_path, TEMP_DIR)
                local_path = os.path.join(TEMP_DIR, file_path)

                # 2. Process
                cap = cv2.VideoCapture(local_path)
                frame_idx = 0
                extracted_this_video = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    if frame_idx % interval == 0:
                        # Resizing for ResNet18
                        resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                        img_name = f"{video_basename}_f{frame_idx}.jpg"
                        cv2.imwrite(os.path.join(dest_folder, img_name), resized_frame)
                        extracted_this_video += 1
                        
                        # Stop immediately if target is hit mid-video
                        if (label == 'fake' and (fake_count + extracted_this_video) >= TARGET_COUNT) or \
                           (label == 'real' and (real_count + extracted_this_video) >= TARGET_COUNT):
                            break
                    frame_idx += 1
                
                cap.release()
                if os.path.exists(local_path): os.remove(local_path)
                
                print(f"🎬 Processed {label}: {video_basename} | Fake Total: {count_existing_images('fake')} | Real Total: {count_existing_images('real')}")

                # Break completely if both targets met
                if count_existing_images('fake') >= TARGET_COUNT and count_existing_images('real') >= TARGET_COUNT:
                    print("🎯 All targets met!")
                    break

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    
    print("\n✨ Done! You can now start training.")

if __name__ == "__main__":
    run_extraction(interval=30)
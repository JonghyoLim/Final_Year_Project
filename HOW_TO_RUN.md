# 🤟 Irish Sign Language Recognition - How to Run

## 📋 **Prerequisites**

- macOS (with Apple Silicon M1/M2/M3/M4)
- Python 3.9+
- Built-in MacBook camera
- ISL_dataset folder with gesture images

---

## 🚀 **Step-by-Step Guide**

### **Step 1: Open Terminal**

1. Press `Cmd + Space`
2. Type "Terminal"
3. Press Enter

---

### **Step 2: Navigate to Project Folder**

```bash
cd ~/Project/Final_Year_Project
```

*(Adjust the path if your project is in a different location)*

---

### **Step 3: Activate Virtual Environment**

```bash
source venv/bin/activate
```

✅ You should see `(venv)` appear at the start of your terminal prompt:
```
(venv) ➜  Final_Year_Project git:(main)
```

---

### **Step 4: Run the Application**

```bash
python standalone_isl_recognition.py
```

---

## ⏱️ **What Happens Next?**

### **First Time Running (One-Time Setup):**

```
============================================================
Irish Sign Language Recognition System
============================================================
No existing model found. Training new model...
Loading dataset...
Loaded 2515 images
Training model...
Epoch 1/100
...
Epoch 100/100
Saving model to ./keras_model.h5
Model loaded successfully!
Starting camera...
```

**⏰ Training Time:** 30-60 minutes (on Apple M4)
- Be patient! This only happens once
- Model will be saved automatically
- Don't close the terminal during training

---

### **Every Time After First Run (Fast Startup):**

```
============================================================
Irish Sign Language Recognition System
============================================================
Loading existing model from ./keras_model.h5
Model loaded successfully!
Starting camera...
Controls:
  - Adjust trackbars to calibrate skin color detection
  - Press ESC to exit
============================================================
```

**⚡ Startup Time:** ~5 seconds
- Loads saved model instantly
- Camera opens immediately

---

## 🎮 **Using the Application**

### **When Camera Opens:**

You'll see **3 windows**:

1. **Camera Output** - Main window with your webcam feed
   - Shows detected letter in RED text
   - Has trackbars for calibration
   - Green rectangle around detected hand

2. **Hand Detection** - Cropped view of your hand

3. **Hand Train (Contour)** - Processed hand image for recognition

---

### **Step 5: Calibrate Skin Color**

Use the **trackbars** in the "Camera Output" window to adjust detection:

- **B for min / G for min / R for min** - Minimum color values
- **B for max / G for max / R for max** - Maximum color values

**🎯 Goal:** Your hand should appear **WHITE** in the skin detection mask

**💡 Tips:**
- Start with default values (already set)
- If hand not detected, increase max values
- If too much background detected, decrease max values
- Lighting matters - sit facing a window or lamp

---

### **Step 6: Make Gestures**

1. **Position your hand** in front of the camera
2. **Hold the gesture steady** for ~10 frames (~0.3 seconds)
3. **Watch for detection:**
   - Terminal prints: `Gesture Detected - Identifying...`
   - Terminal prints: `Detected Letter: A` (or whatever letter)
   - Letter appears in **RED** on camera window

4. **Make another gesture** - repeat!

---

### **Step 7: Exit the Application**

Press **ESC** key to close all windows and stop the camera.

---

## 🔧 **Troubleshooting**

### **Issue: Camera Permission Denied**

**Solution:**
1. Go to **System Preferences** → **Security & Privacy** → **Camera**
2. Check the box next to **Terminal** or **Python**
3. Restart the application

---

### **Issue: Hand Not Detected**

**Symptoms:** No green rectangle around your hand

**Solutions:**
1. **Improve lighting** - face a window or turn on lights
2. **Adjust trackbars** - increase max values for B, G, R
3. **Use plain background** - stand in front of a plain wall
4. **Check distance** - sit 2-3 feet from camera

---

### **Issue: Wrong Letter Detected**

**Causes:**
- Model needs more training data for that gesture
- Hand not held steady enough
- Poor lighting conditions
- Gesture not clear

**Solutions:**
1. Hold gesture more steadily
2. Improve lighting
3. Make gesture more distinct
4. Try again - machine learning isn't 100% accurate

---

### **Issue: "No module named 'tensorflow'"**

**Solution:**
```bash
# Make sure venv is activated (see Step 3)
pip install tensorflow-macos==2.13.0 tensorflow-metal==1.0.1 keras==2.13.1
```

---

### **Issue: Camera Won't Open**

**Symptoms:** Error: "Cannot open camera!"

**Solutions:**
1. Close other apps using the camera (Zoom, FaceTime, etc.)
2. Check camera isn't covered
3. Restart your Mac
4. Try different USB port (if using external camera)

---

### **Issue: Training Takes Too Long**

**Normal:** 30-60 minutes on Apple Silicon
**Too Long:** 2+ hours

**Solutions:**
- Check Activity Monitor - CPU/GPU should be active
- Close other heavy applications
- Let it run overnight if needed
- This is normal for 2515 images and 100 epochs!

---

### **Issue: Want to Retrain the Model**

If you want to start fresh:

```bash
# Delete the saved model
rm keras_model.h5

# Run the application again
python standalone_isl_recognition.py
```

It will train from scratch again.

---

## 💡 **Tips for Best Results**

### **Lighting:**
- ✅ Bright, even lighting from front
- ✅ Face a window during daytime
- ❌ Avoid backlighting (sitting with window behind you)
- ❌ Avoid harsh shadows

### **Background:**
- ✅ Plain wall (white, beige, light color)
- ✅ Consistent background
- ❌ Cluttered background
- ❌ Moving objects behind you

### **Hand Position:**
- ✅ 2-3 feet from camera
- ✅ Hand clearly visible
- ✅ Fingers spread apart for clarity
- ❌ Too close to camera
- ❌ Hand partially out of frame

### **Gesture Quality:**
- ✅ Hold steady for 1-2 seconds
- ✅ Make clear, distinct gestures
- ✅ Practice the ISL alphabet
- ❌ Moving hand while detecting
- ❌ Unclear or rushed gestures

---

## 📁 **Project File Structure**

```
Final_Year_Project/
├── venv/                              # Virtual environment
├── ISL_dataset/                       # Training images
│   ├── 0/
│   ├── 1/
│   ├── ...
│   └── z/
├── standalone_isl_recognition.py      # Main application
├── keras_model.h5                     # Trained model (created after first run)
├── HOW_TO_RUN.md                      # This guide
└── README_INSTALLATION.md             # Installation guide
```

---

## 🔄 **Quick Reference Commands**

```bash
# Navigate to project
cd ~/Project/Final_Year_Project

# Activate virtual environment
source venv/bin/activate

# Run the application
python standalone_isl_recognition.py

# Exit: Press ESC key

# Deactivate virtual environment (when done)
deactivate
```

---

## 🎯 **Expected Behavior**

### **First Run:**
1. ⏳ Training: 30-60 minutes
2. 💾 Saves model: `keras_model.h5`
3. 📹 Opens camera
4. ✅ Ready to recognize gestures!

### **Subsequent Runs:**
1. ⚡ Loads model: ~5 seconds
2. 📹 Opens camera
3. ✅ Ready to recognize gestures!

### **Performance:**
- **Accuracy:** 80-95% (depends on training quality and lighting)
- **Speed:** Real-time detection (~30 FPS)
- **Latency:** ~0.3 seconds from gesture to detection

---

## ❓ **FAQ**

### **Q: Do I need internet to run this?**
**A:** No! Once installed, it runs completely offline.

---

### **Q: Can I use an external webcam?**
**A:** Yes! The script uses camera index 0 (default camera). For external cameras, you might need to change the camera index in the script.

---

### **Q: How accurate is the recognition?**
**A:** Typically 80-95% depending on:
- Training data quality
- Lighting conditions
- How clearly you make gestures
- Background simplicity

---

### **Q: Can I add more gestures?**
**A:** Yes! Add more images to the ISL_dataset folders, delete `keras_model.h5`, and retrain.

---

### **Q: Why does it recognize the wrong letter sometimes?**
**A:** Machine learning isn't perfect. Factors:
- Similar-looking gestures (like 'M' and 'N')
- Poor lighting
- Hand not steady
- Background interference

---

### **Q: Can multiple people use it?**
**A:** Yes! It works with anyone's hands, but different skin tones might need trackbar adjustments.

---

### **Q: Does it work in the dark?**
**A:** No. You need good lighting for the camera to see your hand clearly.

---

### **Q: Can I save the detected letters?**
**A:** Not yet! But you could modify the script to save detected letters to a text file.

---

## 🎓 **Understanding the System**

### **How It Works:**

1. **Camera Capture** → Webcam captures your hand
2. **Skin Detection** → Finds skin-colored regions
3. **Hand Isolation** → Extracts your hand from background
4. **Gesture Stability** → Waits for steady gesture (10 frames)
5. **Image Processing** → Resizes and normalizes hand image
6. **Neural Network** → CNN predicts the letter
7. **Display Result** → Shows letter on screen

---

### **Technologies Used:**

- **OpenCV** → Camera and image processing
- **TensorFlow** → Deep learning framework
- **Keras** → Neural network API
- **CNN** → Convolutional Neural Network architecture
- **NumPy** → Array processing
- **Pillow** → Image manipulation

---

## 📞 **Need Help?**

If you encounter issues:

1. ✅ Check this guide first
2. ✅ Read error messages carefully
3. ✅ Try troubleshooting steps above
4. ✅ Check project README files

---

## 🎉 **You're All Set!**

Enjoy your Irish Sign Language Recognition system! 🤟

**Remember:**
- First run: Be patient (30-60 min training)
- After first run: Lightning fast! ⚡
- Good lighting = Better results 💡
- Steady gestures = Accurate detection ✅

---

**Made with ❤️ for your Final Year Project**

*Last updated: October 27, 2025*
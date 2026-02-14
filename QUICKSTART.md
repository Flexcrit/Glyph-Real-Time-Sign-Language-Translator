# 🚀 Quick Start Guide - ASL Translation System

## How to Run Everything

### **Simple Method: Use the Menu**
```bash
./run_app.sh
```

This will show you an interactive menu with all options:
1. Collect data for one sign
2. Batch collect data for multiple signs  
3. Train the model
4. Run the full app
5. Run camera test
6. Show system info

---

## Step-by-Step Workflow

### **Step 1: Collect Training Data** 📸

#### Option A: Collect One Sign at a Time
```bash
./run_app.sh
# Choose option 1
# Or run directly:
./.venv39/bin/python collect_data.py
```

#### Option B: Batch Collection (Recommended)
```bash
./run_app.sh
# Choose option 2
# Or run directly:
./.venv39/bin/python batch_collect_data.py
```

**Before batch collection:**
1. Edit `batch_collect_data.py`
2. Update the `VOCABULARY` list with your desired signs:
```python
VOCABULARY = [
    "hello",
    "goodbye",
    "thank_you",
    "please",
    # Add your signs here...
]
```

---

### **Step 2: Train the Model** 🏋️

After collecting data for at least 2 signs:
```bash
./run_app.sh
# Choose option 3
# Or run directly:
./.venv39/bin/python train_model.py
```

This will:
- Load all collected data from `training_data/`
- Train an LSTM model
- Save model to `models/asl_model.h5`
- Save labels to `labels.json`
- Display training metrics and graphs

---

### **Step 3: Run the App** 🚀

#### Full App (with predictions):
```bash
./run_app.sh
# Choose option 4
# Or run directly:
./.venv39/bin/streamlit run app.py
```

#### Camera Test (simple):
```bash
./run_app.sh
# Choose option 5
# Or run directly:
./.venv39/bin/streamlit run app_simple.py
```

The app will open in your browser at `http://localhost:8501`

---

## Quick Commands Reference

### Direct Python Commands

```bash
# Data Collection
./.venv39/bin/python collect_data.py          # Single sign
./.venv39/bin/python batch_collect_data.py    # Multiple signs

# Training
./.venv39/bin/python train_model.py

# Running Apps
./.venv39/bin/streamlit run app.py            # Full app
./.venv39/bin/streamlit run app_simple.py     # Camera test
```

### System Info
```bash
./run_app.sh
# Choose option 6
```

Shows:
- Python version
- Installed packages
- Number of signs collected
- Model status
- Labels status

---

## Troubleshooting

### "Permission denied" when running run_app.sh
```bash
chmod +x run_app.sh
```

### "No module named 'tensorflow'" or similar
```bash
./.venv39/bin/pip install -r requirements.txt
```

### Camera not working
- Check camera permissions in System Preferences → Security & Privacy → Camera
- Try different camera index in app settings (0, 1, or 2)

### App opens but no predictions
- Make sure you've trained a model first (Step 2)
- Check that `models/asl_model.h5` exists
- Check that `labels.json` exists

---

## Complete Workflow Example

```bash
# 1. Edit vocabulary
nano batch_collect_data.py  # Add your signs to VOCABULARY list

# 2. Collect data
./run_app.sh  # Choose option 2
# Perform each sign 30 times when prompted

# 3. Train model
./run_app.sh  # Choose option 3
# Wait for training to complete (~5-10 minutes)

# 4. Run app
./run_app.sh  # Choose option 4
# App opens in browser, start signing!

# 5. Test and iterate
# If accuracy is low, collect more data or add more signs
```

---

## File Structure

After completing the workflow:
```
sign-language-detector/
├── run_app.sh                 # Main runner (use this!)
├── collect_data.py            # Single sign collection
├── batch_collect_data.py      # Batch collection (new!)
├── train_model.py             # Model training
├── app.py                     # Full Streamlit app
├── app_simple.py              # Simple camera test
├── training_data/             # Your collected data
│   ├── 0000_hello/
│   ├── 0001_goodbye/
│   └── ...
├── models/
│   └── asl_model.h5          # Trained model
└── labels.json               # Sign labels mapping
```

---

## Next Steps

1. ✅ **Start small**: Collect 3-5 signs first to test the system
2. ✅ **Train and verify**: Make sure the model works before collecting more
3. ✅ **Scale up**: Use `batch_collect_data.py` to add more signs
4. ✅ **Refer to README_SCALING.md** for advanced options (pre-trained datasets, optimization)

---

**Need help?** Check:
- `README_SCALING.md` - How to scale to 100s or 1000s of signs
- Run `./run_app.sh` and choose option 6 for system diagnostics

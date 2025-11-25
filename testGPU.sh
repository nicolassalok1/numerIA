python3 - << 'EOF'
import torch, tensorflow as tf, sklearn, subprocess

print("\n========================")
print("🔍 TEST GPU GLOBAL")
print("========================")

# --- NVIDIA-SMI ---
print("\n📌 nvidia-smi:")
try:
    print(subprocess.check_output("nvidia-smi", shell=True, text=True))
except Exception as e:
    print("nvidia-smi ERROR:", e)

# --- PyTorch ---
print("\n📌 PyTorch:")
print("Version :", torch.__version__)
print("CUDA disponible :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))

# --- TensorFlow ---
print("\n📌 TensorFlow:")
print("Version :", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPU détecté :", len(gpus) > 0)
print("Liste des GPU :", gpus)

# --- Scikit-Learn ---
print("\n📌 Scikit-Learn:")
print("Version :", sklearn.__version__)

print("\n========================")
print("✔ TEST TERMINÉ")
print("========================\n")
EOF

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, label_binarize
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_align_data(seq_path, track_path):
    seq_data = np.load(seq_path, allow_pickle=True)
    track_data = np.load(track_path, allow_pickle=True)

    X_seq, y_seq, track_ids_seq = seq_data['X'], seq_data['y'], seq_data['track_ids']
    X_track, y_track, track_ids_track = track_data['X'], track_data['y'], track_data['track_ids']

    if X_seq.shape[1] == 11 and X_seq.shape[2] == 20:
        print("transposing...")
        X_seq = np.transpose(X_seq, (0, 2, 1))

    track_id_to_index = {
        tuple(tid) if isinstance(tid, (list, tuple, np.ndarray)) else (tid,): i
        for i, tid in enumerate(track_ids_track)
    }

    X_seq_matched, X_track_matched, y_matched = [], [], []
    for i, tid in enumerate(track_ids_seq):
        key = tuple(tid) if isinstance(tid, (list, tuple, np.ndarray)) else (tid,)
        if key in track_id_to_index:
            idx = track_id_to_index[key]
            X_seq_matched.append(X_seq[i])
            X_track_matched.append(X_track[idx])
            y_matched.append(y_seq[i])

    print(f"[DEBUG] Matched pairs: {len(X_seq_matched)}")
    return np.array(X_seq_matched), np.array(X_track_matched), np.array(y_matched)


class UnifiedFusionModel(nn.Module):
    def __init__(self, seq_input_size, track_input_size, hidden_size=64, dropout=0.0):
        super().__init__()
        self.track_input_size = track_input_size

        self.lstm = nn.LSTM(input_size=seq_input_size, hidden_size=hidden_size,
                            batch_first=True, bidirectional=True)

        if track_input_size > 0:
            self.track_fc = nn.Sequential(
                nn.Linear(track_input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.use_track = True
        else:
            self.use_track = False

        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_size * (2 + int(self.use_track)), 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x_seq, x_track):
        lstm_out, _ = self.lstm(x_seq)
        lstm_feat = lstm_out[:, -1, :]
        if self.use_track:
            track_feat = self.track_fc(x_track)
            fused = torch.cat([lstm_feat, track_feat], dim=1)
        else:
            fused = lstm_feat
        return self.fusion_fc(fused)

from sklearn.utils.class_weight import compute_class_weight

def Train_UnifiedFusionModel(seq_path, track_path, model_save_path, result_path,
                             seq_input_size=9, track_input_size=12, hidden_size=64, dropout=0.0):
    print("[STEP 1] Loading and aligning data...")
    X_seq, X_track, y = load_and_align_data(seq_path, track_path)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    class_weights_np = compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)

    X_seq_train, X_seq_test, X_track_train, X_track_test, y_train, y_test = train_test_split(
        X_seq, X_track, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    X_seq_train = torch.tensor(X_seq_train, dtype=torch.float32)
    X_seq_test = torch.tensor(X_seq_test, dtype=torch.float32)
    X_track_train = torch.tensor(X_track_train, dtype=torch.float32)
    X_track_test = torch.tensor(X_track_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_seq_train, X_track_train, y_train_tensor)
    test_dataset = TensorDataset(X_seq_test, X_track_test, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = UnifiedFusionModel(seq_input_size=seq_input_size, track_input_size=track_input_size,
                               hidden_size=hidden_size, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("[STEP 2] Training unified fusion model...")
    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch_seq, batch_track, batch_y in train_loader:
            batch_seq, batch_track, batch_y = batch_seq.to(device), batch_track.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_seq, batch_track)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_loader):.4f}")

    print("[STEP 3] Evaluating...")
    model.eval()
    with torch.no_grad():
        logits = model(X_seq_test.to(device), X_track_test.to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    acc = np.mean(preds == y_test)
    f1 = f1_score(y_test, preds, average="macro")

    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    try:
        auc = roc_auc_score(y_test_bin, probs, average="macro", multi_class="ovo")
    except:
        auc = -1

    print("[RESULT] Accuracy:", acc)
    print(classification_report(y_test, preds, target_names=[str(cls) for cls in le.classes_]))

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("Unified Fusion Model Confusion Matrix")
    os.makedirs(result_path, exist_ok=True)
    plt.savefig(os.path.join(result_path, "confusion_matrix.png"))
    plt.close()

    torch.save(model.state_dict(), model_save_path)
    print("Model saved to", model_save_path)

    return {
        "accuracy": acc,
        "f1_score": f1,
        "auc": auc,
        "confusion_matrix": cm.tolist()
    }

def Test_UnifiedFusionModel(seq_path, track_path, model_save_path):
    print("[STEP 1] Loading and aligning data...")
    X_seq, X_track, y = load_and_align_data(seq_path, track_path)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32)
    X_track_tensor = torch.tensor(X_track, dtype=torch.float32)

    model = UnifiedFusionModel(seq_input_size=X_seq.shape[2], track_input_size=X_track.shape[1]).to(device)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(X_seq_tensor.to(device), X_track_tensor.to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    acc = np.mean(preds == y_encoded)
    f1 = f1_score(y_encoded, preds, average="macro")
    
    y_encoded_bin = label_binarize(y_encoded, classes=np.unique(y_encoded))
    auc = roc_auc_score(y_encoded_bin, probs, average="macro", multi_class="ovo")
    print("[RESULT] Accuracy:", acc)
    print(classification_report(y_encoded, preds, target_names=[str(cls) for cls in le.classes_]))
    cm = confusion_matrix(y_encoded, preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    print("[RESULT] AUC:", auc)
    return {
        "accuracy": acc,
        "f1_score": f1,
        "auc": auc,
        "confusion_matrix": cm.tolist()
    }
if __name__ == "__main__":
    from Config import GENERATED_DIR, SEQ_LEN, MODEL_DIR, SEQ_RESULT_DIR
    SEQ_DATA_PATH = f"{GENERATED_DIR}/trajectory_dataset_{SEQ_LEN}.npz"
    TRACK_DATA_PATH = f"{GENERATED_DIR}/track_dataset.npz"
    MODEL_SAVE_PATH = f"{MODEL_DIR}/unified_fusion_model.pth"
    os.makedirs(SEQ_RESULT_DIR, exist_ok=True)
    Train_UnifiedFusionModel(SEQ_DATA_PATH, TRACK_DATA_PATH, MODEL_SAVE_PATH, SEQ_RESULT_DIR)
if __name__ == "__main__":
    from Config import UNI_RESULT_DIR, SEQ_LEN, MODEL_DIR, SEQ_RESULT_DIR, features, track_features, GENERATED_DIR
    SEQ_DATA_PATH = f"{GENERATED_DIR}/trajectory_dataset_{SEQ_LEN}.npz"
    TRACK_DATA_PATH = f"{GENERATED_DIR}/track_dataset.npz"
    MODEL_SAVE_PATH = f"{MODEL_DIR}/unified_fusion_model.pth"
    os.makedirs(UNI_RESULT_DIR, exist_ok=True)
    Test_UnifiedFusionModel(SEQ_DATA_PATH, TRACK_DATA_PATH, MODEL_SAVE_PATH)

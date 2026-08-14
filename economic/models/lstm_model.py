import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
import json

PROC_DIR   = Path("economic/data/processed")
MODELS_DIR = Path("economic/models/saved")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS     = 100
LR         = 0.001
PATIENCE   = 15    # early stopping patience

print(f"Device: {DEVICE}")

# ── Dataset ───────────────────────────────────────────────────────

class EconomicDataset(Dataset):
    """
    PyTorch Dataset wrapping numpy arrays.

    WHY a Dataset class?
      PyTorch's DataLoader needs a Dataset to:
        - Know how many samples exist (__len__)
        - Retrieve one sample at a time (__getitem__)
        - Batch samples automatically
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)  # shape: (n, 60, 2)
        self.y = torch.FloatTensor(y)  # shape: (n,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Model ─────────────────────────────────────────────────────────

class EconomicLSTM(nn.Module):
    """
    2-layer LSTM for time series regression.

    WHY LSTM over vanilla RNN?
      LSTM has a "memory cell" (c_t) controlled by 3 gates:
        - Forget gate:  how much of past memory to keep
        - Input gate:   how much of new info to store
        - Output gate:  what to output from memory

      This lets LSTM learn LONG-TERM dependencies —
      e.g. "USD/MYR weakness 30 days ago still matters now"
      — which vanilla RNN forgets due to vanishing gradients.

    Architecture choices:
      hidden_size=64: balance between capacity and overfitting
                      (128+ would overfit on ~3000 samples)
      num_layers=2:   stack two LSTM layers for more abstraction
                      (layer 1 = short patterns, layer 2 = longer)
      dropout=0.2:    randomly zero 20% of neurons during training
                      (regularization — prevents memorizing training data)
    """

    def __init__(self, input_size=2, hidden_size=64,
                 num_layers=2, dropout=0.2):
        super(EconomicLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Core LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,          # between LSTM layers (not after last)
            batch_first=True          # input shape: (batch, seq, features)
        )

        # Dropout after LSTM (before linear)
        self.dropout = nn.Dropout(dropout)

        # Final linear layer: hidden_size → 1 prediction
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass.

        x shape: (batch_size, seq_len, input_size)
                  e.g. (32, 60, 2)

        h0, c0: initial hidden state + cell state
                zeros = "no prior memory at start of each batch"

        lstm_out: (batch, seq_len, hidden_size)
                  output at EVERY timestep

        lstm_out[:, -1, :]: take ONLY the LAST timestep's output
                  this is the LSTM's "summary" of the entire sequence
                  — used to predict the NEXT value
        """
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).to(DEVICE)

        lstm_out, _ = self.lstm(x, (h0, c0))

        # Take last timestep only
        last_out = lstm_out[:, -1, :]      # (batch, hidden_size)
        last_out = self.dropout(last_out)

        prediction = self.fc(last_out)     # (batch, 1)
        return prediction.squeeze(1)       # (batch,)


# ── Training ──────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    """One full pass through training data"""
    model.train()
    total_loss = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()          # clear previous gradients
        predictions = model(X_batch)   # forward pass
        loss = criterion(predictions, y_batch)  # compute MSE loss
        loss.backward()                # backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
        optimizer.step()               # update weights

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion):
    """Evaluate on validation set (no gradient computation)"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()

    return total_loss / len(loader)


def train_model(target: str = "klci"):
    """Full training loop with early stopping"""
    print(f"\n{'='*55}")
    print(f"Training LSTM for: {target.upper()}")
    print(f"{'='*55}")

    # Load data
    X_train = np.load(PROC_DIR / f"X_train_{target}.npy")
    y_train = np.load(PROC_DIR / f"y_train_{target}.npy")
    X_val   = np.load(PROC_DIR / f"X_val_{target}.npy")
    y_val   = np.load(PROC_DIR / f"y_val_{target}.npy")

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # DataLoaders
    train_loader = DataLoader(
        EconomicDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        EconomicDataset(X_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False
    )

    # Model, loss, optimizer
    model     = EconomicLSTM().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Learning rate scheduler: reduce LR if val loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=7, factor=0.5
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for up to {EPOCHS} epochs (early stop patience={PATIENCE})\n")

    # Training loop with early stopping
    best_val_loss  = float('inf')
    patience_count = 0
    history        = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss   = validate(model, val_loader, criterion)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{EPOCHS} | "
                  f"Train loss: {train_loss:.6f} | "
                  f"Val loss: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(),
                       MODELS_DIR / f"lstm_{target}_best.pt")
        else:
            patience_count += 1

        # Early stopping
        if patience_count >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            print(f"Best val loss: {best_val_loss:.6f}")
            break

    # Save training history
    with open(MODELS_DIR / f"history_{target}.json", "w") as f:
        json.dump(history, f)

    print(f"\nModel saved: economic/models/saved/lstm_{target}_best.pt")
    return model, history, best_val_loss


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}

    for target in ["klci", "usd_myr"]:
        model, history, best_val_loss = train_model(target)
        results[target] = {
            "best_val_loss": best_val_loss,
            "epochs_trained": len(history['train_loss'])
        }

    print(f"\n{'='*55}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*55}")
    for target, r in results.items():
        print(f"  {target:<10}: best_val_loss={r['best_val_loss']:.6f} "
              f"({r['epochs_trained']} epochs)")

    print(f"\nNext: python economic/models/evaluator.py")
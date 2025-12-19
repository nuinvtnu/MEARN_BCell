

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(x + self.dropout(attn_output))
        return x

class ESM_Attn_ResidualNet(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=256, num_heads=4):
        super(ESM_Attn_ResidualNet, self).__init__()
        self.attn = MultiHeadAttention(input_dim, num_heads)
        self.residual_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, input_dim)
        )
        self.norm = nn.LayerNorm(input_dim)
        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.attn(x).squeeze(1)
        x = self.norm(x + self.residual_mlp(x))
        return self.output_layer(x).squeeze(1)

class EpitopeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    return running_loss / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_x", required=True)
    parser.add_argument("--train_y", required=True)
    parser.add_argument("--output_dir", default="model_output")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pos_weight", type=float, default=5.0)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    X_train = np.load(args.train_x, allow_pickle=True)
    y_train = np.load(args.train_y, allow_pickle=True)

    train_loader = DataLoader(EpitopeDataset(X_train, y_train),
                              batch_size=args.batch_size, shuffle=True)

    model = ESM_Attn_ResidualNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=device))

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}/{args.epochs}, Loss: {loss:.4f}")

    final_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save({"model_state_dict": model.state_dict()}, final_path)
    print("Saved:", final_path)

if __name__ == "__main__":
    main()

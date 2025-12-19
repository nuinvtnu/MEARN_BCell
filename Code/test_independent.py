def evaluate(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int().cpu().numpy()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    auc_score = roc_auc_score(all_labels, all_probs)
    bac = balanced_accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    prec, rec, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(rec, prec)

    TN, FP, FN, TP = cm.ravel()
    SN = TP / (TP + FN + 1e-6)
    SP = TN / (TN + FP + 1e-6)

    return {
        "cm": cm, "ACC": acc, "AUC": auc_score, "BAC": bac,
        "MCC": mcc, "PR_AUC": pr_auc, "SN": SN, "SP": SP
    }

def print_metrics(m, title):
    print("\n====", title, "====")
    print("Confusion Matrix:", m["cm"])
    print(f"ACC: {m['ACC']:.4f}")
    print(f"AUC: {m['AUC']:.4f}")
    print(f"BAC: {m['BAC']:.4f}")
    print(f"MCC: {m['MCC']:.4f}")
    print(f"PR-AUC: {m['PR_AUC']:.4f}")
    print(f"SN: {m['SN']:.4f}")
    print(f"SP: {m['SP']:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_x", required=True)
    parser.add_argument("--train_y", required=True)
    parser.add_argument("--test_x", required=True)
    parser.add_argument("--test_y", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pos_weight", type=float, default=5.0)
    parser.add_argument("--model_out", default="final_full_train.pth")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    X_train = np.load(args.train_x, allow_pickle=True)
    y_train = np.load(args.train_y, allow_pickle=True)
    X_test = np.load(args.test_x, allow_pickle=True)
    y_test = np.load(args.test_y, allow_pickle=True)

    train_loader = DataLoader(EpitopeDataset(X_train, y_train),
                              batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(EpitopeDataset(X_test, y_test),
                             batch_size=args.batch_size)

    model = ESM_Attn_ResidualNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=device))

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}/{args.epochs}, Loss: {loss:.4f}")

    torch.save({"model_state_dict": model.state_dict()}, args.model_out)
    print("Saved:", args.model_out)

    metrics = evaluate(model, test_loader, device)
    print_metrics(metrics, "Independent Test")

if __name__ == "__main__":
    main()

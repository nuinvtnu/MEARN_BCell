def main():
    parser = argparse.ArgumentParser(description="MEARN Pipeline Controller")

    parser.add_argument("--mode", required=True,
                        choices=["train", "cross", "test"],
                        help="train = train model only, cross = cross-validation, test = train_full + independent test")

    # generic arguments (forwarded to sub-modules)
    parser.add_argument("--train_x")
    parser.add_argument("--train_y")
    parser.add_argument("--test_x")
    parser.add_argument("--test_y")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--pos_weight", type=float)
    parser.add_argument("--output_dir")
    parser.add_argument("--model_out")
    parser.add_argument("--n_splits", type=int)
    parser.add_argument("--no_cuda", action="store_true")

    args, unknown = parser.parse_known_args()

    # Forward remaining args to submodule by rewriting sys.argv
    # Keep program name of the submodule as argv[0]
    forwarded = []
    for k, v in vars(args).items():
        if v is None:
            continue
        if k == "mode":
            continue
        if isinstance(v, bool):
            if v:
                forwarded += [f"--{k}"]
        else:
            forwarded += [f"--{k}", str(v)]

    if args.mode == "train":
        print("\\n===== Running built_model.py (Train Only) =====")
        sys.argv = ["built_model.py"] + forwarded
        built_model_main()

    elif args.mode == "cross":
        print("\\n===== Running cross_validation.py =====")
        sys.argv = ["cross_validation.py"] + forwarded
        cross_main()

    elif args.mode == "test":
        print("\\n===== Running test_independent.py =====")
        sys.argv = ["test_independent.py"] + forwarded
        test_main()

if __name__ == "__main__":
    main()

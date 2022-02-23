def add_exp_opts(parser):
    parser.add_argument(
        "--exp_id",
        default="models/handnet_exp",
        type=str,
        help="Path of current experiment (default:debug)",
    )
    parser.add_argument(
        "--host_folder",
        default="models/handnet_exp/host_folder",
        type=str,
        help="Path to folder where to save plotly train/validation curves",
    )
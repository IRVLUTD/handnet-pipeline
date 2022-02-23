def add_nets3d_opts(parser):
    # Options for mano prevision and supervision
    parser.add_argument(
        "--hidden_neurons",
        nargs="+",
        default=[1024, 256],
        type=int,
        help="Number of neurons in hidden layer for mano decoder",
    )
    parser.add_argument(
        "--mano_use_shape",
        action="store_false",
        help="Predict MANO shape parameters",
    )
    parser.add_argument("--mano_use_joints2d", action="store_false", help="Predict 2D joints")
    parser.add_argument(
        "--mano_use_pca",
        action="store_false",
        help="Predict pca parameters directly instead of rotation angles",
    )
    parser.add_argument(
        "--mano_comps",
        choices=list(range(5, 46)),
        default=45,
        type=int,
        help="Number of PCA components",
    )

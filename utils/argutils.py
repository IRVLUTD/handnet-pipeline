def parse_general_args(parser):
    # General Args
    parser.add_argument('--device', default='cuda', help='device')
    # resume trained model
    parser.add_argument('-r', '--resume', default='', help='resume from checkpoint')

def parse_e2e_args(parser):
    # General Args
    parser.add_argument('--device', default='cuda', help='device')
    # resume trained model
    parser.add_argument('--detect_resume', default='', help='resume from detector checkpoint')
    parser.add_argument('--mano_resume', default='', help='resume from detector checkpoint')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--model_name', help='directory to save models', required=True, type=str)
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument('--net', dest='net',
                    help='res18, res34, res50, res101, res152, fcos, e2e', 
                    default='e2e', type=str)


    # Detector options
    parser.add_argument('-det_bs', '--det_batch_size', default=2, type=int)
    parser.add_argument('--det_epochs', default=25, type=int, metavar='N',
                        help='number of total detector epochs to run')
    parser.add_argument('--det_lr', default=0.00125, type=float, help='initial detector learning rate')
    parser.add_argument('--det_momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--det_optimizer', default='sgd', type=str, help='optimizer')
    parser.add_argument('--det_wd', '--det_weight_decay', default=1e-4, type=float,
                        metavar='W', help='detector weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument(
        "--det_lr_steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument('--det_lr_gamma', default=0.1, type=float, help='decrease lr by a factor of det_lr_gamma')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument("--det_amp", action="store_true", help="Use torch.cuda.amp for mixed precision detector training")

    # MANO options
    parser.add_argument('--mano_bs', '--mano_batch_size', default=32, type=int)
    parser.add_argument('--mano_epochs', default=35, type=int, metavar='N',
                        help='number of total mano epochs to run')
    parser.add_argument('--mano_lr', default=1e-3, type=float, help='initial mano learning rate')
    parser.add_argument('--mano_momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--mano_optimizer', default='sgd', type=str, help='optimizer')
    parser.add_argument('--mano_wd', '--mano_weight_decay', default=0, type=float,
                        metavar='W', help='mano weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument(
        "--mano_lr_steps",
        default=30,
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument('--mano_lr_gamma', default=0.1, type=float, help='decrease lr by a factor of mano_lr_gamma')


    args = parser.parse_args()

    return args


def parse_a2j_args(parser):
    parser.add_argument('--lr-step-size', default=10, type=int)

def parse_training_args(parser):
    import os

    # Detector options
    parser.add_argument('--net', dest='net',
                    help='res18, res34, res50, res101, res152, fcos, e2e, a2j', 
                    default='e2e', type=str)
    parser.add_argument('-bs', '--batch-size', default=1, type=int)
    parser.add_argument('--epochs', default=25, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)


    parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)

    # set detector training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # save model and log
    parser.add_argument('--model_name',
                        help='directory to save models', required=True, type=str)

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    args = parser.parse_args()

    return args
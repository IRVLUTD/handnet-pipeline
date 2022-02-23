def parse_general_args(parser):
    # General Args
    parser.add_argument('--device', default='cuda', help='device')
    # resume trained model
    parser.add_argument('-r', '--resume', default='', help='resume from checkpoint')


def parse_3d_args(parser):
    from mano_train.options.nets3dopts import add_nets3d_opts
    add_nets3d_opts(parser)

    # use pretrained fcos model
    parser.add_argument('--pretrained', default='models/fcos_dexycb_res34/detector_1_8.pth', help='resume pre-trained fcos model from checkpoint')




def parse_detection_args(parser):
    import os

    # Detector options
    parser.add_argument('--net', dest='net',
                    help='res18, res34, res50, res101, res152, fcos, e2e', 
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
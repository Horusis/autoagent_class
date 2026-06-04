import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8,9"

import argparse
from run_overall import main_offline

def get_args_parser():
    parser = argparse.ArgumentParser(description="A demo for our project")
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--server_port", type=int, default=None, 
                        help=("will start gradio app on this port (if available). If None, will search for an available port starting at 7860."))
    parser.add_argument("--viser_server_port", type=int, default=8080, 
                        help="will start viser server on this port (if available), default is 8080")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--tmp_dir", type=str, default="./tmp", help="value for tempfile.tempdir")
    parser.add_argument("--xsam_dir", type=str, default="X-SAM", help="value for xsam root directory")
    parser.add_argument("--per_gpu", type=bool, default=True, help="XSam and SLAM3R will use the same GPU if False, otherwise they will use different GPUs")

    return parser

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    # try:
    print("Start the app")
    main_offline(parser)
    # except Exception as e:
    #     os.removedirs(args.tmp_dir)
        
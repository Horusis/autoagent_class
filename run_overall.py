import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8,9"
import sys
import argparse
import gradio
import os.path as osp
import tempfile
import functools

from mmengine.config import Config
from xtuner.tools.utils import set_model_resource
from xsam.utils.logging import set_default_logging_format
from xsam.utils.utils import register_function
from xsam.demo.demo_part import XSamDemo
import warnings

from slam3r.models import Local2WorldModel, Image2PointsModel

from app_components import *

import torch

set_default_logging_format()
warnings.filterwarnings("ignore")


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



def main_demo(XSam, sam2_predictor, i2p_model, l2w_model, device, tmpdirname, server_name, server_port, per_gpu):
    initial_df = info_to_df(CUSTOM_SEGMENTS_INFO)
    initial_target_choices = [x["name"] for x in CUSTOM_SEGMENTS_INFO]
    initial_target = initial_target_choices[0] if initial_target_choices else None
    initial_prompt = 'sem: ' + ', '.join([f"{info['name']}" for info in CUSTOM_SEGMENTS_INFO])

    segment_scene_func = functools.partial(segment_scene, XSam, sam2_predictor, per_gpu)
    recon_scene_func = functools.partial(recon_scene, i2p_model, l2w_model, device, per_gpu)

    with gradio.Blocks(css=""".gradio-container {margin: 0 !important; min-width: 100%};""", title="Extract Volume Demo") as demo:
        # scene state is save so that you can change num_points_save... without rerunning the inference
        per_frame_res = gradio.State(None)
        tmpdir_name = gradio.State(tmpdirname)

        # Initialization of states
        segment_info_state = gradio.State(copy.deepcopy(CUSTOM_SEGMENTS_INFO))
        xsam_output_files_update_state = gradio.State(False)
        xsam_active_state = gradio.State([])
        xsam_removed_state = gradio.State([])

        segmentation_result_state = gradio.State(None)
        instance_info = gradio.State(None)

        target_instances_state = gradio.State([])
        base_height_state = gradio.State(1.0)

        gradio.HTML('<h2 style="text-align: center;">Extract Volume Demo</h2>')

        with gradio.Column():
            with gradio.Row():
                input_type = gradio.Dropdown(["directory", "images", "video"], scale=1, value='images', label="select type of input files")
                video_extract_fps = gradio.Number(value=5, scale=0, interactive=True, visible=False, label="fps for extracting frames from video")
                inputfiles = gradio.File(file_count="multiple", file_types=["image"], scale=2, height=200, label="Select a directory containing images")

                image_gallery = gradio.Gallery(label="Click or use the left/right arrow keys to browse images", visible=True, selected_index=0, preview=False, height=300, scale=2, buttons=[])
                video_gallery = gradio.Video(label="Uploaded Video", visible=False, height=300, scale=2, buttons=[])

            with gradio.Row():
                # XSam prompt (segment target selected by users)
                with gradio.Column(scale=2):
                    segment_table = gradio.Dataframe(
                        value=initial_df,
                        headers=["name"],
                        datatype=["str"],
                        row_count=(len(initial_df), "dynamic"),
                        col_count=(1, "fixed"),
                        interactive=True,
                        label="Segment Names",
                        max_height=300,
                    )
                with gradio.Column(scale=1):
                    segment_label_input = gradio.Textbox(label="Segment Label", interactive=True)
                    add_row_btn = gradio.Button("Add Segment Label")
                    segmented_labels = gradio.Textbox(value=None, label="Segmented Labels", interactive=False)

                with gradio.Column(scale=2):
                    XSam_prompt = gradio.Textbox(value=initial_prompt, label="prompt for segmentation", interactive=False)

            with gradio.Row():
                # XSam settings
                XSam_vprompt_mask = gradio.Textbox(value=None, label="visual prompt for segmentation (Not implemented)", interactive=False)
                XSam_threshold = gradio.Slider(value=0.3, minimum=0., maximum=1., interactive=True, label="confidence threshold for segmentation")
                XSam_background_labels = gradio.Textbox(value='wall, floor, ceiling-merged', label="label for background in the prompt", interactive=True)

            XSam_run_btn = gradio.Button("Run X-SAM", interactive=True)

            with gradio.Row():
                # XSam result visualization
                XSam_removed_output_files = gradio.File(file_count="multiple", file_types=["image"], value=[], scale=2, height=200, visible=True, interactive=True, label="X-SAM Output Files which not used for SLAM3R")
                XSam_removed_output_gallery = gradio.Gallery(label="X-SAM Output which not used for SLAM3R", visible=True, interactive=False, selected_index=0, preview=False, height=300, scale=2, buttons=[])
                XSam_active_output_files = gradio.File(file_count="multiple", file_types=["image"], value=[], scale=2, height=200, visible=True, interactive=True, label="X-SAM Output Files which Input of SLAM3R")
                XSam_active_output_gallery = gradio.Gallery(label="X-SAM Output", visible=True, interactive=False, selected_index=0, preview=False, height=300, scale=2, buttons=[])

            with gradio.Row():
                # SLAM3R settings
                kf_stride = gradio.Dropdown(["auto", "manual setting"], label="how to choose stride between keyframes", value='manual setting', interactive=True, info="For I2P reconstruction!")
                kf_stride_fix = gradio.Slider(value=-1, minimum=-1, maximum=-1, step=1, visible=True, interactive=False, label="stride between keyframes\n(for manual setting)", info="For I2P reconstruction!")
                win_r = gradio.Number(value=2, precision=0, minimum=1, maximum=200, interactive=True, label="the radius of the input window", info="For I2P reconstruction!")
                initial_winsize = gradio.Number(value=2, precision=0, minimum=2, maximum=200, interactive=True, label="the number of frames for initialization", info="For I2P reconstruction!")
                conf_thres_i2p = gradio.Slider(value=1.5, minimum=1., maximum=10, interactive=True, label="confidence threshold for the i2p model", info="For I2P reconstruction!")

            with gradio.Row():
                # SLAM3R settings
                num_scene_frame = gradio.Slider(value=1, minimum=1., maximum=100, step=1, interactive=True, label="the number of scene frames for reference", info="For L2W reconstruction!")
                buffer_strategy = gradio.Dropdown(["reservoir", "fifo", "unbounded"], value='reservoir', interactive=True, label="strategy for buffer management", info="For L2W reconstruction!")
                buffer_size = gradio.Number(value=100, precision=0, minimum=1, interactive=True, visible=True, label="size of the buffering set", info="For L2W reconstruction!")
                update_buffer_intv = gradio.Number(value=1, precision=0, minimum=1, interactive=True, label="the interval of updating the buffering set", info="For L2W reconstruction!")

            slam3r_run_btn = gradio.Button("Run SLAM3R", interactive=True)

            with gradio.Row():
                with gradio.Column(scale=3):
                    out_3d_model = gradio.Model3D(height=500, clear_color=(0., 0., 0., 0.3), interactive=False)
                with gradio.Column(scale=1):
                    # adjust the confidence threshold
                    conf_thres_l2w = gradio.Slider(value=4, minimum=1., maximum=100, interactive=False, label="confidence threshold for the result", )
                    # adjust the camera size in the output pointcloud
                    num_points_save = gradio.Number(value=1000000, precision=0, minimum=1, interactive=False, label="number of points sampled from the result", )

            with gradio.Row():
                with gradio.Column(scale=2):
                    base_instance_dbox = gradio.Dropdown(
                        choices=None,
                        value="",
                        interactive=True,
                        label="Base Target",
                    )
                with gradio.Column(scale=2):
                    target_list_radio = gradio.Radio(choices=[], value=None, info="Show point cloud of target", interactive=False)
                with gradio.Column(scale=2):
                    null_radio = gradio.Radio(choices=[], info="Null space for better visualization", interactive=False, visible=False)

            with gradio.Row():
                # visualize each pcd segmented by X-SAM
                with gradio.Column(scale=2):
                    base_pcd_3d_model = gradio.Model3D(height=300, label="Base target point cloud", clear_color=(0., 0., 0., 0.3), interactive=False)
                    base_caption = gradio.HTML('<div style="min-height: 10px; text-align: center;">base target</div>')
                    base_height = gradio.Textbox(label="Height (meter)", value=None, interactive=False)
                with gradio.Column(scale=2):
                    target_pcd_3d_model = gradio.Model3D(height=300, label="Target point cloud", clear_color=(0., 0., 0., 0.3), interactive=False)
                    target_caption = gradio.HTML('<div style="min-height: 10px; text-align: center;">main target 1</div>')
                    target_height = gradio.Textbox(label="Height (meter)", value=None, interactive=False)
                    target_width = gradio.Textbox(label="Width (meter)", value=None, interactive=False)
                    target_depth = gradio.Textbox(label="Depth (meter)", value=None, interactive=False)
                    target_bbox_volume = gradio.Textbox(label="Box Volume (cubic meter)", value=None, interactive=False)
                with gradio.Column(scale=2):
                    real_height = gradio.Number(value=2.1, precision=2, minimum=0.1, interactive=False, label="real height of the base target (meter)", )
                    std_interval_multiplier = gradio.Number(value=1, precision=1, minimum=0.1, interactive=False, label="the multiplier for standard deviation when segmenting the point")
                    # adjust the cluster settings
                    clustering_before_alignment = gradio.Checkbox(label="Apply clustering before alignment for better visualization", value=True, interactive=False)
                    volume_clustering = gradio.Checkbox(label="Apply clustering for volume estimation", value=True, interactive=False)
                    cluster_eps = gradio.Slider(value=0.02, minimum=0.005, maximum=1.0, step=0.005, interactive=False, label="eps for DBSCAN clustering (meter)", )
                    cluster_min_points = gradio.Slider(value=10, minimum=1, step=1, interactive=False, label="min points for DBSCAN clustering", )

            # XSam group each target
            XSam_toggle_output_group = [XSam_run_btn]
            XSam_sync_input_group = [XSam_active_output_files, XSam_removed_output_files, xsam_output_files_update_state, xsam_active_state, xsam_removed_state]
            XSam_sync_output_group = [XSam_active_output_files, XSam_removed_output_files, XSam_active_output_gallery, XSam_removed_output_gallery, xsam_active_state, xsam_removed_state]
            unlock_function_after_xsam_group = [slam3r_run_btn]

            # SLAM3R group
            slam3r_toggle_output_group = [slam3r_run_btn, conf_thres_l2w, num_points_save, real_height, cluster_eps, cluster_min_points]
            scene_update_input_group = [tmpdir_name, base_instance_dbox, segmentation_result_state, instance_info, num_points_save, conf_thres_l2w, real_height, std_interval_multiplier, clustering_before_alignment, volume_clustering, cluster_eps, cluster_min_points]
            base_group = [base_pcd_3d_model, base_caption, base_height]
            target_group = [target_pcd_3d_model, target_caption, target_height, target_width, target_depth, target_bbox_volume]
            unlock_function_after_slam3r_group = [conf_thres_l2w, num_points_save, base_instance_dbox, target_list_radio, real_height, clustering_before_alignment, volume_clustering, cluster_eps, cluster_min_points]

            # XSAM events
            inputfiles.change(display_inputs, inputs=[inputfiles, input_type], outputs=[image_gallery, video_gallery], queue=False)
            input_type.change(change_inputfile_type, inputs=[input_type], outputs=[inputfiles, video_extract_fps, image_gallery, video_gallery], queue=False)
            segment_table.change(fn=sync_ui, inputs=[segment_table, segment_info_state], outputs=[segment_table, segment_info_state, XSam_prompt], queue=False)
            add_row_btn.click(fn=add_segment_row, inputs=[segment_table, segment_label_input, segment_info_state], outputs=[segment_table, segment_info_state, XSam_prompt, segment_label_input], queue=False)
            XSam_start = XSam_run_btn.click(fn=disable_contents, inputs=XSam_toggle_output_group, outputs=XSam_toggle_output_group)
            XSam_run = XSam_start.then(fn=segment_scene_func, inputs=[inputfiles, video_extract_fps, segment_info_state, XSam_vprompt_mask, XSam_threshold, XSam_background_labels, tmpdir_name], outputs=[*XSam_sync_output_group, segment_info_state, segmented_labels, segmentation_result_state, instance_info, base_instance_dbox])
            XSam_end = XSam_run.then(fn=enable_contents, inputs=XSam_toggle_output_group, outputs=XSam_toggle_output_group)
            XSam_end.then(fn=enable_contents, inputs=unlock_function_after_xsam_group, outputs=unlock_function_after_xsam_group)

            gradio.on(triggers=[XSam_active_output_files.change, XSam_removed_output_files.change], fn=on_output_files_changed, inputs=XSam_sync_input_group, outputs=[*XSam_sync_output_group, xsam_output_files_update_state])

            # SLAM3R events
            kf_stride.change(change_kf_stride_type, inputs=[kf_stride], outputs=[kf_stride_fix], queue=False)
            buffer_strategy.change(change_buffer_strategy, inputs=[buffer_strategy], outputs=[buffer_size], queue=False)

            slam3r_start = slam3r_run_btn.click(fn=disable_contents, inputs=slam3r_toggle_output_group, outputs=slam3r_toggle_output_group)

            slam3r_run = slam3r_start.then(fn=recon_scene_func, inputs=[XSam_active_output_files, kf_stride_fix, win_r, initial_winsize, conf_thres_i2p, num_scene_frame, update_buffer_intv, buffer_strategy, buffer_size, *scene_update_input_group], outputs=[out_3d_model, *base_group, *target_group, target_list_radio, target_instances_state, per_frame_res])
            slam3r_end = slam3r_run.then(fn=enable_contents, inputs=slam3r_toggle_output_group, outputs=slam3r_toggle_output_group)
            slam3r_end.then(fn=enable_contents, inputs=unlock_function_after_slam3r_group, outputs=unlock_function_after_slam3r_group)

            conf_thres_l2w.release(fn=get_model_from_scene, inputs=[per_frame_res, *scene_update_input_group], outputs=[out_3d_model, *base_group, *target_group, target_list_radio, target_instances_state])
            num_points_save.change(fn=get_model_from_scene, inputs=[per_frame_res, *scene_update_input_group], outputs=[out_3d_model, *base_group, *target_group, target_list_radio, target_instances_state])

            real_height.change(fn=get_model_from_scene, inputs=[per_frame_res, *scene_update_input_group], outputs=[out_3d_model, *base_group, *target_group, target_list_radio, target_instances_state])
            base_instance_dbox.change(fn=get_model_from_scene, inputs=[per_frame_res, *scene_update_input_group], outputs=[out_3d_model, *base_group, *target_group, target_list_radio, target_instances_state])
            std_interval_multiplier.change(fn=get_model_from_scene, inputs=[per_frame_res, *scene_update_input_group], outputs=[out_3d_model, *base_group, *target_group, target_list_radio, target_instances_state])
            clustering_before_alignment.change(fn=get_model_from_scene, inputs=[per_frame_res, *scene_update_input_group], outputs=[out_3d_model, *base_group, *target_group, target_list_radio, target_instances_state])
            volume_clustering.change(fn=get_model_from_scene, inputs=[per_frame_res, *scene_update_input_group], outputs=[out_3d_model, *base_group, *target_group, target_list_radio, target_instances_state])
            cluster_eps.change(fn=get_model_from_scene, inputs=[per_frame_res, *scene_update_input_group], outputs=[out_3d_model, *base_group, *target_group, target_list_radio, target_instances_state])
            cluster_min_points.change(fn=get_model_from_scene, inputs=[per_frame_res, *scene_update_input_group], outputs=[out_3d_model, *base_group, *target_group, target_list_radio, target_instances_state])

            target_list_radio.change(fn=change_target_visualization, inputs=[target_list_radio, target_instances_state], outputs=target_group, queue=False)

    return demo



# def main_offline(parser: argparse.ArgumentParser):
#     args = parser.parse_args()
parser = get_args_parser()
args = parser.parse_args()

if args.tmp_dir is not None:
    tmp_path = args.tmp_dir
    os.makedirs(tmp_path, exist_ok=True)
    tempfile.tempdir = tmp_path

if args.server_name is not None:
    server_name = args.server_name
else:
    server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

# X-SAM
# Create XSam instance
if args.xsam_dir is not None and os.path.exists(args.xsam_dir):
    os.environ['root_dir'] = os.path.join(os.getcwd(), args.xsam_dir)

if gradio.NO_RELOAD:
    cfg = Config.fromfile(osp.expanduser(osp.expandvars('$root_dir/xsam/xsam/configs/xsam/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune.py')))
    set_model_resource(cfg)
    register_function(cfg._cfg_dict)
    pth_model = osp.expanduser(osp.expandvars('$root_dir/inits/X-SAM/s3_mixed_finetune/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune/pytorch_model.bin'))

    if args.per_gpu:
        torch.cuda.set_device(0)
    XSam = XSamDemo(cfg, pth_model, output_ids_with_output=False)

    base_dir = os.getcwd()
    sam2_dir = base_dir + "/sam2"
    sys.path.insert(0, sam2_dir)
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")
    # from sam2.build_sam import build_sam2_video_predictor
    # checkpoint = sam2_dir + "/checkpoints/sam2.1_hiera_large.pt"
    # model_cfg = sam2_dir + "/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    # predictor = build_sam2_video_predictor(model_cfg, checkpoint)

    sys.path.remove(sam2_dir)


    if args.per_gpu:
        torch.cuda.set_device(1)
    # SLAM3R
    i2p_model = Image2PointsModel.from_pretrained('siyan824/slam3r_i2p')
    l2w_model = Local2WorldModel.from_pretrained('siyan824/slam3r_l2w')
    i2p_model.to(args.device)
    l2w_model.to(args.device)
    i2p_model.eval()
    l2w_model.eval()

# extract volume demo will write the 3D model inside tmpdirname
# with tempfile.TemporaryDirectory(suffix='extract_volume_demo') as tmpdirname:
    # main_demo(XSam, i2p_model, l2w_model, args.device, tmpdirname, server_name, args.server_port, args.per_gpu)

    tmpdir_obj = tempfile.TemporaryDirectory(suffix='extract_volume_demo')
    tmpdirname = tmpdir_obj.name
demo = main_demo(XSam, predictor, i2p_model, l2w_model, args.device, tmpdirname, server_name, args.server_port, args.per_gpu)



if __name__ == "__main__":
    # main_offline(argparse.ArgumentParser())
    demo.launch(share=False, server_name=server_name, server_port=args.server_port)
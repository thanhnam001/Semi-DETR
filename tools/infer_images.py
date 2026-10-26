import argparse
import glob
import os
import os.path as osp

import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector
from mmdet.models import build_detector

from detr_ssod.utils import patch_config

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Semi-DETR on images")
    parser.add_argument("config", help="config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("input", help="an image file or a directory of images")
    parser.add_argument("--out-dir", default="vis_results", help="where painted images go")
    parser.add_argument("--out-json", help="optional json file to dump raw detections")
    parser.add_argument("--score-thr", type=float, default=0.3, help="bbox score threshold")
    parser.add_argument("--device", default="cuda:0", help="device used for inference")
    parser.add_argument(
        "--model", choices=["teacher", "student"], default="teacher",
        help="which sub-model to run (tools/test.py uses teacher)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if "semi_wrapper" in cfg:
        cfg.semi_wrapper.test_cfg.inference_on = args.model
    cfg = patch_config(cfg)
    # mmdet's init_detector injects `pretrained=None`, which DinoDetrSSOD rejects,
    # so build the model the same way tools/test.py does
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES") or cfg.get("classes")
    model.cfg = cfg  # inference_detector reads the test pipeline from here
    # the head defaults to the one-to-one (post warm-up) branch when no curr_step is given
    model.to(args.device)
    model.eval()

    if osp.isdir(args.input):
        files = sorted(
            f for f in glob.glob(osp.join(args.input, "**", "*"), recursive=True)
            if f.lower().endswith(IMG_EXTS)
        )
    else:
        files = [args.input]

    mmcv.mkdir_or_exist(args.out_dir)
    dump = []
    prog_bar = mmcv.ProgressBar(len(files))
    for f in files:
        result = inference_detector(model, f)
        rel = osp.relpath(f, args.input) if osp.isdir(args.input) else osp.basename(f)
        model.show_result(
            f, result, score_thr=args.score_thr, out_file=osp.join(args.out_dir, rel)
        )
        if args.out_json:
            dets = []
            for cls_id, bboxes in enumerate(result):
                for x1, y1, x2, y2, score in bboxes.tolist():
                    if score >= args.score_thr:
                        dets.append(dict(
                            label=model.CLASSES[cls_id], bbox=[x1, y1, x2, y2], score=score
                        ))
            dump.append(dict(file=rel, detections=dets))
        prog_bar.update()

    if args.out_json:
        mmcv.dump(dump, args.out_json)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()

import argparse
import glob
import json
import os.path as osp
import shutil

import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector
from mmdet.models import build_detector

from detr_ssod.utils import patch_config

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
LABELME_VERSION = "4.5.7"


def dump_labelme(img_path, json_dir, result, classes, width, height, score_thr):
    """Write <stem>.json next to a copy of the image in `json_dir` (LabelMe pairs by stem)."""
    mmcv.mkdir_or_exist(json_dir)
    name = osp.basename(img_path)
    if osp.realpath(osp.dirname(img_path)) != osp.realpath(json_dir):
        shutil.copy2(img_path, osp.join(json_dir, name))

    shapes = []
    for cls_id, bboxes in enumerate(result):
        for x1, y1, x2, y2, score in bboxes.tolist():
            if score < score_thr:
                continue
            x1, x2 = min(max(x1, 0.0), width), min(max(x2, 0.0), width)
            y1, y2 = min(max(y1, 0.0), height), min(max(y2, 0.0), height)
            if x2 <= x1 or y2 <= y1:
                continue
            shapes.append(dict(
                label=str(classes[cls_id]),
                points=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                group_id=None,
                shape_type="polygon",
                flags=dict(score=round(score, 4)),
            ))

    data = dict(
        version=LABELME_VERSION,
        flags={},
        imagePath=name,
        imageData=None,
        imageHeight=int(height),
        imageWidth=int(width),
        clockwise_rotated_angle=0,
        shapes=shapes,
    )
    json_path = osp.join(json_dir, osp.splitext(name)[0] + ".json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Semi-DETR on images")
    parser.add_argument("config", help="config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("input", help="an image file or a directory of images")
    parser.add_argument("--out-dir", help="optional dir for painted images")
    parser.add_argument(
        "--labelme-dir",
        help="dump one LabelMe json per image here (images are copied alongside); "
        "pass the input dir itself to write the jsons next to the originals",
    )
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

    assert args.out_dir or args.labelme_dir, "specify --out-dir and/or --labelme-dir"
    prog_bar = mmcv.ProgressBar(len(files))
    for f in files:
        img = mmcv.imread(f)
        height, width = img.shape[:2]
        result = inference_detector(model, img)  # boxes are in original-image pixels
        rel = osp.relpath(f, args.input) if osp.isdir(args.input) else osp.basename(f)
        if args.out_dir:
            model.show_result(
                img, result, score_thr=args.score_thr, out_file=osp.join(args.out_dir, rel)
            )
        if args.labelme_dir:
            dump_labelme(
                f, osp.join(args.labelme_dir, osp.dirname(rel)), result,
                model.CLASSES, width, height, args.score_thr,
            )
        prog_bar.update()


if __name__ == "__main__":
    main()

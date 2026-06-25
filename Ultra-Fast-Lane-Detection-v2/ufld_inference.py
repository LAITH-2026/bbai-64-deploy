"""
Shared UFLDv2 inference pieces (no DALI): ParsingNet, anchors, pred2coords.

Used by:
  - infer_image.py (CLI one-shot)
  - lane_service/inference_runtime.py (HTTP service)
"""
import numpy as np
import torch

from utils.config import Config
from model.backbone import resnet


def pred2coords(pred, row_anchor, col_anchor, local_width=1, original_image_width=1640, original_image_height=590):
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred["loc_row"].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred["loc_col"].shape

    max_indices_row = pred["loc_row"].argmax(1).cpu()
    valid_row = pred["exist_row"].argmax(1).cpu()
    max_indices_col = pred["loc_col"].argmax(1).cpu()
    valid_col = pred["exist_col"].argmax(1).cpu()

    pred["loc_row"] = pred["loc_row"].cpu()
    pred["loc_col"] = pred["loc_col"].cpu()

    coords = []
    row_lane_idx = [1, 2]
    col_lane_idx = [0, 3]

    for i in row_lane_idx:
        tmp = []
        if valid_row[0, :, i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    lo = max(0, max_indices_row[0, k, i] - local_width)
                    hi = min(num_grid_row - 1, max_indices_row[0, k, i] + local_width) + 1
                    all_ind = torch.tensor(list(range(lo, hi)))
                    out_tmp = (pred["loc_row"][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row - 1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0, :, i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    lo = max(0, max_indices_col[0, k, i] - local_width)
                    hi = min(num_grid_col - 1, max_indices_col[0, k, i] + local_width) + 1
                    all_ind = torch.tensor(list(range(lo, hi)))
                    out_tmp = (pred["loc_col"][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_col - 1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)

    return coords


def initialize_weights(module: torch.nn.Module) -> None:
    """
    Lightweight reimplementation of the project's initialize_weights
    that does NOT import training/DALI utilities.
    """
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)


class ParsingNet(torch.nn.Module):
    """
    Minimal CULane/TuSimple head copied from model/model_culane.py,
    but without auxiliary segmentation head or any utils.common imports.
    """

    def __init__(
        self,
        backbone: str,
        num_grid_row: int,
        num_cls_row: int,
        num_grid_col: int,
        num_cls_col: int,
        num_lane_on_row: int,
        num_lane_on_col: int,
        input_height: int,
        input_width: int,
        fc_norm: bool = False,
    ) -> None:
        super().__init__()
        self.num_grid_row = num_grid_row
        self.num_cls_row = num_cls_row
        self.num_grid_col = num_grid_col
        self.num_cls_col = num_cls_col
        self.num_lane_on_row = num_lane_on_row
        self.num_lane_on_col = num_lane_on_col

        self.dim1 = self.num_grid_row * self.num_cls_row * self.num_lane_on_row
        self.dim2 = self.num_grid_col * self.num_cls_col * self.num_lane_on_col
        self.dim3 = 2 * self.num_cls_row * self.num_lane_on_row
        self.dim4 = 2 * self.num_cls_col * self.num_lane_on_col
        self.total_dim = self.dim1 + self.dim2 + self.dim3 + self.dim4

        mlp_mid_dim = 2048
        self.input_dim = input_height // 32 * input_width // 32 * 8

        self.model = resnet(backbone, pretrained=True)

        self.cls = torch.nn.Sequential(
            torch.nn.LayerNorm(self.input_dim) if fc_norm else torch.nn.Identity(),
            torch.nn.Linear(self.input_dim, mlp_mid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_mid_dim, self.total_dim),
        )
        self.pool = (
            torch.nn.Conv2d(512, 8, 1)
            if backbone in ["34", "18", "34fca"]
            else torch.nn.Conv2d(2048, 8, 1)
        )

        initialize_weights(self.cls)

    def forward(self, x):
        x2, x3, fea = self.model(x)
        fea = self.pool(fea)
        fea = fea.view(-1, self.input_dim)
        out = self.cls(fea)

        pred_dict = {
            "loc_row": out[:, : self.dim1].view(
                -1, self.num_grid_row, self.num_cls_row, self.num_lane_on_row
            ),
            "loc_col": out[:, self.dim1 : self.dim1 + self.dim2].view(
                -1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col
            ),
            "exist_row": out[
                :, self.dim1 + self.dim2 : self.dim1 + self.dim2 + self.dim3
            ].view(-1, 2, self.num_cls_row, self.num_lane_on_row),
            "exist_col": out[:, -self.dim4 :].view(
                -1, 2, self.num_cls_col, self.num_lane_on_col
            ),
        }
        return pred_dict


def build_cfg(config_path):
    cfg = Config.fromfile(config_path)
    if cfg.dataset == "CULane":
        cfg.row_anchor = np.linspace(0.42, 1, cfg.num_row)
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    elif cfg.dataset == "Tusimple":
        cfg.row_anchor = np.linspace(160, 710, cfg.num_row) / 720
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    else:
        raise ValueError(
            "Only CULane or Tusimple configs are supported (same as demo.py)."
        )
    return cfg

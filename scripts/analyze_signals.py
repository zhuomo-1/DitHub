#!/usr/bin/env python
"""Analyze which signals best separate object vs background queries in frozen GD."""
import sys, os, torch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from detectron2.config import LazyConfig, instantiate
from groundingdino.util.task_memory import TaskMemory
from groundingdino.config.configs.common.data.odinw.mapping_classes import ODINW_13_FILE_MAPPING
from main_nsps import load_grounding_dino, _set_memory_classes_for_batch
from groundingdino.util.misc import NestedTensor, nested_tensor_from_tensor_list
from groundingdino.models.GroundingDINO.bertwarper import generate_masks_with_special_tokens_and_transfer_map
import torch.nn.functional as F
from groundingdino.util.misc import inverse_sigmoid
from torchvision.ops import box_iou

device = torch.device('cuda')
TaskMemory().task_mapping = ODINW_13_FILE_MAPPING
gd = load_grounding_dino(
    'groundingdino/config/GroundingDINO_SwinT_OGC_dt_ncrn.py',
    'groundingdino_swint_ogc.pth', '/tmp', device
)


def run_gd_forward(gd, batch, device):
    images = gd.preprocess_image(batch)
    samples = nested_tensor_from_tensor_list(images)
    captions = [x['captions'] for x in batch]
    names_list = [x['captions'][:-1].split('.') for x in batch]
    _set_memory_classes_for_batch(batch, names_list)
    tokenized = gd.tokenizer(captions, padding='longest', return_tensors='pt').to(device)
    text_self_attention_masks, position_ids, _ = (
        generate_masks_with_special_tokens_and_transfer_map(
            tokenized, gd.specical_tokens, gd.tokenizer
        )
    )
    if text_self_attention_masks.shape[1] > gd.max_text_len:
        L = gd.max_text_len
        text_self_attention_masks = text_self_attention_masks[:, :L, :L]
        position_ids = position_ids[:, :L]
        tokenized['input_ids'] = tokenized['input_ids'][:, :L]
        tokenized['attention_mask'] = tokenized['attention_mask'][:, :L]
        tokenized['token_type_ids'] = tokenized['token_type_ids'][:, :L]
    if gd.sub_sentence_present:
        enc_input = {k: v for k, v in tokenized.items() if k != 'attention_mask'}
        enc_input['attention_mask'] = text_self_attention_masks
        enc_input['position_ids'] = position_ids
    else:
        enc_input = tokenized
    bert_out = gd.bert(**enc_input)
    encoded_text = gd.feat_map(bert_out['last_hidden_state'])
    text_token_mask = tokenized.attention_mask.bool()
    if encoded_text.shape[1] > gd.max_text_len:
        L = gd.max_text_len
        encoded_text = encoded_text[:, :L, :]
        text_token_mask = text_token_mask[:, :L]
        position_ids = position_ids[:, :L]
        text_self_attention_masks = text_self_attention_masks[:, :L, :L]
    text_dict = {
        'encoded_text': encoded_text, 'text_token_mask': text_token_mask,
        'position_ids': position_ids, 'text_self_attention_masks': text_self_attention_masks,
    }
    features, poss = gd.backbone(samples)
    srcs, masks = [], []
    for l, feat in enumerate(features):
        src, mask = feat.decompose()
        srcs.append(gd.input_proj[l](src)); masks.append(mask)
    if gd.num_feature_levels > len(srcs):
        _len_srcs = len(srcs)
        for l in range(_len_srcs, gd.num_feature_levels):
            src = gd.input_proj[l](features[-1].tensors if l == _len_srcs else srcs[-1])
            m = samples.mask
            mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
            pos_l = gd.backbone[1](NestedTensor(src, mask)).to(src.dtype)
            srcs.append(src); masks.append(mask); poss.append(pos_l)
    hs, ref, _, _, _ = gd.transformer(srcs, masks, None, poss, None, None, text_dict)
    return hs, ref, images, text_dict


tasks = [
    'test/test_odinw13/for_train/test_AerialMaritimeDrone_tiled.py',
    'test/test_odinw13/for_train/test_CottontailRabbits.py',
    'test/test_odinw13/for_train/test_Egohands_generic.py',
]

for task_path in tasks:
    task_name = os.path.basename(task_path).replace('.py', '').replace('test_', '')
    cfg = LazyConfig.load(task_path)
    cfg.dataloader.train.num_workers = 0
    loader = instantiate(cfg.dataloader.train)

    sig_data = {k: {'pos': [], 'neg': []} for k in
                ['q_norm', 'area', 'ref_shift', 'box_consistency', 'gd_objectness']}

    for bi, batch in enumerate(loader):
        if bi >= 10:
            break
        with torch.no_grad():
            hs, ref, images, text_dict = run_gd_forward(gd, batch, device)

        for i in range(len(batch)):
            if 'instances' not in batch[i] or len(batch[i]['instances']) == 0:
                continue

            hs_last = hs[-1][i]
            last_ref = ref[-2][i]
            delta = gd.bbox_embed[-1](hs_last.unsqueeze(0))[0]
            boxes = (delta + inverse_sigmoid(last_ref)).sigmoid()

            q_norms = hs_last.norm(dim=-1)
            area = boxes[:, 2] * boxes[:, 3]
            init_ref = ref[0][i]
            ref_shift = (last_ref - init_ref[:, :last_ref.shape[-1]]).norm(dim=-1)

            all_boxes_layers = []
            for li in range(len(hs)):
                h_l = hs[li][i]
                r_l = ref[li][i] if li < len(ref) else ref[-1][i]
                d_l = gd.bbox_embed[li](h_l.unsqueeze(0))[0]
                b_l = (d_l + inverse_sigmoid(r_l)).sigmoid()
                all_boxes_layers.append(b_l)
            box_stack = torch.stack(all_boxes_layers)
            box_var = box_stack.var(dim=0).sum(dim=-1)
            box_consistency = 1.0 / (box_var + 1e-4)

            gd_logits = gd.class_embed[-1](hs_last.unsqueeze(0), text_dict)
            gd_obj = gd_logits.max(dim=-1).values.sigmoid()[0]

            gt_boxes_raw = batch[i]['instances'].gt_boxes.tensor.to(device)
            img_h, img_w = images.image_sizes[i]
            gt_norm = torch.zeros(len(gt_boxes_raw), 4, device=device)
            gt_norm[:, 0] = gt_boxes_raw[:, 0] / img_w
            gt_norm[:, 1] = gt_boxes_raw[:, 1] / img_h
            gt_norm[:, 2] = gt_boxes_raw[:, 2] / img_w
            gt_norm[:, 3] = gt_boxes_raw[:, 3] / img_h

            pred_xyxy = torch.zeros(boxes.shape[0], 4, device=device)
            pred_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
            pred_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
            pred_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
            pred_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

            iou_mat = box_iou(pred_xyxy, gt_norm)
            max_iou = iou_mat.max(dim=1).values
            is_pos = max_iou > 0.5
            is_neg = max_iou < 0.05

            for name, sig in [('q_norm', q_norms), ('area', area),
                              ('ref_shift', ref_shift),
                              ('box_consistency', box_consistency),
                              ('gd_objectness', gd_obj)]:
                sig_data[name]['pos'].append(sig[is_pos].cpu())
                sig_data[name]['neg'].append(sig[is_neg].cpu())

    print(f'\n=== {task_name} (10 train imgs) ===')
    for name, data in sig_data.items():
        pos = torch.cat(data['pos']) if data['pos'] else torch.tensor([])
        neg = torch.cat(data['neg']) if data['neg'] else torch.tensor([])
        if len(pos) > 0 and len(neg) > 0:
            neg_med = neg.median()
            sep = (pos > neg_med).float().mean().item()
            print(f'  {name:20s}: pos_mean={pos.mean():.4f}  neg_mean={neg.mean():.4f}  sep={sep:.0%}')
        else:
            print(f'  {name:20s}: insufficient data (pos={len(pos)}, neg={len(neg)})')

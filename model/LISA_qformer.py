import torch
import torch.nn as nn
from typing import List, Union, Tuple, Optional
import torch.nn.functional as F

from model.segment_anything import build_sam_vit_h
from model.llava.model.language_model.llava_llama import (
    LlavaLlamaForCausalLM,
    LlavaLlamaModel,
)

from utils.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX


def calculate_dice_loss(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    mask_count: float,
    scale_factor=1000,
    epsilon=1e-6,
):
    """
    Calculate the DICE loss, a measure similar to generalized IOU for masks.
    """
    predictions = predictions.sigmoid()
    predictions = predictions.flatten(1, 2)
    ground_truth = ground_truth.flatten(1, 2)

    intersection = 2 * (predictions / scale_factor * ground_truth).sum(dim=-1)
    union = (predictions / scale_factor).sum(dim=-1) + (
        ground_truth / scale_factor
    ).sum(dim=-1)

    dice_loss = 1 - (intersection + epsilon) / (union + epsilon)
    dice_loss = dice_loss.sum() / (mask_count + 1e-8)
    return dice_loss


def compute_sigmoid_cross_entropy(
    predictions: torch.Tensor, targets: torch.Tensor, mask_count: float
):
    """
    Compute sigmoid cross-entropy loss for binary classification.
    """
    loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1)
    loss = loss.sum() / (mask_count + 1e-8)
    return loss


class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs):
        self._set_model_configurations(config, kwargs)
        super().__init__(config)
        self.model = LisaModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _set_model_configurations(self, config, kwargs):
        config.mm_use_image_start_end = kwargs.pop("use_mm_start_end", True)
        config.mm_vision_module = kwargs.get(
            "vision_module", "openai/clip-vit-large-patch14"
        )
        self._initialize_loss_weights(kwargs)
        config.bbox_token_idx = kwargs.get("bbox_token_idx", 1)
        config.num_reg_features = kwargs.get("num_level_reg_features", 4)
        config.with_region = kwargs.get("with_region", False)
        config.bbox_token_idx = kwargs.get("bbox_token_idx", 32002)
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.img_emb_len = kwargs.pop("img_emb_len")
        self.seg_image_tokens = kwargs.pop("seg_image_tokens")
        self.max_num_images = kwargs.pop("max_num_images")

    def _initialize_loss_weights(self, kwargs):
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)

    def get_grounding_encoder_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            return torch.cat(
                [self._encode_single_image(img) for img in pixel_values], dim=0
            )

    def _encode_single_image(self, image):
        torch.cuda.empty_cache()
        return self.model.visual_model.image_encoder(image.unsqueeze(0))

    def forward(self, **kwargs):
        return (
            super().forward(**kwargs)
            if "past_key_values" in kwargs
            else self.model_forward(**kwargs)
        )

    def model_forward(
        self,
        global_enc_images: torch.FloatTensor,
        grounding_enc_images: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        qformer_input_ids: torch.LongTensor,
        qformer_attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        image_offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        image_paths: List[str],
        inference: bool = False,
        **kwargs,
    ):
        # Extract grounding encoder image embeddings
        if grounding_enc_images is not None:
            image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
            assert image_embeddings.shape[0] == image_offset[-1]

        seg_token_masks = self._create_seg_token_mask_multi(
            input_ids, qformer_attention_masks, infer=inference
        )

        # if not seg_token_masks:
        #     seg_token_masks = [
        #         self._create_seg_token_mask_multi(
        #             input_ids, qformer_attention_masks, seg_only=True, infer=inference
        #         )
        #     ]

        conversation_list = kwargs.get("conversation_list")
        print(f"inside {conversation_list = }")
        print(f"inside {len(seg_token_masks) = }")
        for idx, seg_token_mask in enumerate(seg_token_masks):
            print(f"first {idx}: {seg_token_mask = }")
        print("")
        for idx, seg_token_mask in enumerate(seg_token_masks):
            if isinstance(seg_token_mask, torch.Tensor):
                print(f"{idx}: {seg_token_mask.shape = }")
            else:
                print(f"{idx}: {len(seg_token_mask) = }")

        # Handle inference or training paths
        if inference:
            output_hidden_states = self._inference_path(
                input_ids,
                global_enc_images,
                attention_masks,
                image_offset,
            )
        else:
            output, output_hidden_states = self._training_path(
                global_enc_images,
                input_ids,
                labels,
                attention_masks,
                offset,
                image_offset,
            )

        # Process hidden states
        hidden_states, pred_embeddings = self._process_hidden_states_multi(
            output_hidden_states, seg_token_masks, offset
        )
        # Generate and post-process masks
        pred_masks = self._generate_and_postprocess_masks_multi(
            pred_embeddings, image_embeddings, resize_list, label_list, image_offset
        )

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": masks_list,
            }

        # Calculate losses
        return self._calculate_losses(pred_masks, masks_list, output, image_paths)

    # def _create_seg_token_mask(self, input_ids, infer=False):
    #     mask = input_ids[:, 1:] == self.seg_token_idx
    #     if not infer:
    #         return torch.cat(
    #             [
    #                 torch.zeros(
    #                     (
    #                         mask.shape[0],
    #                         (self.img_emb_len - 1),
    #                     ),
    #                     dtype=torch.bool,
    #                     device=input_ids.device,
    #                 ),
    #                 mask,
    #                 torch.zeros(
    #                     (mask.shape[0], 1), dtype=torch.bool, device=input_ids.device
    #                 ),
    #             ],
    #             dim=1,
    #         )
    #     return torch.cat(
    #         [
    #             torch.zeros(
    #                 (
    #                     mask.shape[0],
    #                     (self.img_emb_len - 1),
    #                 ),
    #                 dtype=torch.bool,
    #                 device=input_ids.device,
    #             ),
    #             mask,
    #         ],
    #         dim=1,
    #     )

    def _create_seg_token_mask(self, input_ids, qformer_attention_masks, infer=False):
        mask = input_ids[:, 1:] == self.seg_token_idx
        if not infer:
            return torch.cat(
                [
                    torch.zeros(
                        (
                            mask.shape[0],
                            (self.img_emb_len - 1),
                        ),
                        dtype=torch.bool,
                        device=input_ids.device,
                    ),
                    mask,
                    torch.zeros(
                        (mask.shape[0], 1), dtype=torch.bool, device=input_ids.device
                    ),
                ],
                dim=1,
            )
        return torch.cat(
            [
                torch.zeros(
                    (
                        mask.shape[0],
                        (self.img_emb_len - 1),
                    ),
                    dtype=torch.bool,
                    device=input_ids.device,
                ),
                mask,
            ],
            dim=1,
        )

    def _create_shifted_mask(self, input_ids, shift, token_idx):
        bsz = input_ids.shape[0]
        padding = shift - 1
        new_mask = input_ids[:, shift:] == token_idx
        if padding == 0:
            return new_mask
        return torch.cat(
            [
                new_mask,
                torch.zeros((bsz, padding), dtype=torch.bool, device=input_ids.device),
            ],
            dim=1,
        )

    def _create_seg_token_mask_multi(
        self, input_ids, qformer_attention_masks, seg_only=False, infer=False
    ):
        image_idx = 0
        all_masks = []
        bsz = input_ids.shape[0]
        for image_idx in range(len(self.seg_image_tokens[: self.max_num_images])):
            mask = torch.ones_like(input_ids[:, 1:], device=input_ids.device).bool()
            if not seg_only:
                for idx, token_idx in enumerate(self.seg_image_tokens[image_idx]):
                    shifted_mask = self._create_shifted_mask(
                        input_ids, idx + 1, token_idx
                    )
                    mask &= shifted_mask
            all_masks.append(mask)

        seg_only_mask = torch.ones_like(
            input_ids[:, 1:], device=input_ids.device
        ).bool()
        for mask in all_masks:
            seg_only_mask &= ~mask

        all_masks = [seg_only_mask, *all_masks]

        img_count = (input_ids == IMAGE_TOKEN_INDEX).sum(dim=1)
        # qformer_query_token_count = qformer_attention_masks.sum(dim=1)

        all_new_masks = []
        for image_idx, masks in enumerate(all_masks):
            new_masks = []
            for batch_idx in range(bsz):
                mask = masks[batch_idx]
                mask = torch.cat(
                    [
                        torch.zeros(
                            (
                                self.img_emb_len
                                # + qformer_query_token_count[batch_idx]
                                - 1
                            )
                            * img_count[batch_idx],
                            device=input_ids.device,
                        ).bool(),
                        mask,
                    ],
                    dim=0,
                )
                if not infer:
                    mask = torch.cat(
                        (mask, torch.zeros(1, device=input_ids.device).bool()), dim=0
                    )
                new_masks.append(mask)

            if any(x.shape != new_masks[0].shape for x in new_masks):
                max_len = max(x.shape[0] for x in new_masks)

                new_masks_align = []
                for cur_new_mask in new_masks:
                    cur_new_mask = torch.cat(
                        (
                            cur_new_mask,
                            torch.zeros(
                                (
                                    max_len
                                    - cur_new_mask.shape[0]  # ,
                                    # cur_new_mask.shape[1],
                                ),
                                dtype=cur_new_mask.dtype,
                                device=cur_new_mask.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_masks_align.append(cur_new_mask)
                new_masks = new_masks_align
            new_masks = torch.stack(new_masks, dim=0)
            all_new_masks.append(new_masks)

        while len(all_new_masks) > 0 and torch.all(all_new_masks[-1] == False):
            all_new_masks.pop()

        return all_new_masks

    def _inference_path(
        self,
        input_ids,
        global_enc_images,
        attention_masks,
        image_offset,
    ):
        length = input_ids.shape[0]
        global_enc_images_extended = global_enc_images.unsqueeze(0)
        if length > 1:
            global_enc_images_extended = global_enc_images_extended.repeat(
                length, 1, 1, 1, 1
            )

        # Process and return inference output
        output_hidden_states = []
        for i in range(input_ids.shape[0]):
            output_i = super().forward(
                images=global_enc_images_extended[i : i + 1].squeeze(0),
                attention_mask=attention_masks[i : i + 1],
                input_ids=input_ids[i : i + 1],
                output_hidden_states=True,
            )
            output_hidden_states.append(output_i.hidden_states)
            torch.cuda.empty_cache()

        output_hidden_states = torch.cat(output_hidden_states, dim=0)
        output_hidden_states = [output_hidden_states]
        return output_hidden_states

    def _training_path(
        self,
        global_enc_images,
        input_ids,
        labels,
        attention_masks,
        offset,
        image_offset,
    ):
        if global_enc_images is not None:
            global_enc_images, image_offset = self._prepare_global_enc_image(
                global_enc_images, offset, image_offset
            )

        print(f"inside {global_enc_images.shape = }")

        output = super().forward(
            images=global_enc_images,
            attention_mask=attention_masks,
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
        )
        output_hidden_states = output.hidden_states
        return output, output_hidden_states

    def _prepare_global_enc_image(self, global_enc_image, offset, image_offset):
        global_enc_image_list = []
        new_image_offset = [0]
        img_cnt = 0
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            img_start_i, img_end_i = image_offset[i].item(), image_offset[i + 1].item()
            global_enc_image_i = global_enc_image[img_start_i:img_end_i].contiguous()
            if end_i - start_i > 1:
                global_enc_image_i = global_enc_image_i.repeat(end_i - start_i, 1, 1, 1)
            global_enc_image_list.append(global_enc_image_i)
            for _ in range(end_i - start_i):
                new_image_offset.append(img_cnt := img_cnt + (img_end_i - img_start_i))
        return torch.cat(global_enc_image_list, dim=0), torch.as_tensor(
            new_image_offset, device=image_offset.device
        )

    def _process_hidden_states(
        self, output_hidden_states, seg_token_mask, offset, infer=False
    ):
        hidden_states = [self.model.text_hidden_fcs[0](output_hidden_states[-1])]
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1, device=last_hidden_state.device).long(), seg_token_offset],
            dim=0,
        )
        if not infer:
            seg_token_offset = seg_token_offset[offset]

        pred_embeddings_list = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_list.append(pred_embeddings[start_i:end_i])
        return hidden_states, pred_embeddings_list

    def _process_hidden_states_multi(
        self, output_hidden_states, seg_token_masks, offset, infer=False
    ):
        hidden_states = [self.model.text_hidden_fcs[0](output_hidden_states[-1])]
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        print(f"{last_hidden_state.shape = }")
        all_pred_embeddings_lists = []
        # for img_idx in range(2):
        for img_idx in range(len(seg_token_masks)):
            pred_embeddings = last_hidden_state[seg_token_masks[img_idx]]
            seg_token_counts = seg_token_masks[img_idx].int().sum(-1)
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [
                    torch.zeros(1, device=seg_token_offset.device).long(),
                    seg_token_offset,
                ],
                dim=0,
            )
            if not infer:
                seg_token_offset = seg_token_offset[offset]

            pred_embeddings_list = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                if start_i == end_i:
                    pred_embeddings_list.append(None)
                    continue
                pred_embeddings_list.append(pred_embeddings[start_i:end_i])
            all_pred_embeddings_lists.append(pred_embeddings_list)
        return hidden_states, all_pred_embeddings_lists  # [[], [seg, seg seg]]

    def _generate_and_postprocess_masks(
        self, pred_embeddings, image_embeddings, resize_list, label_list, infer=False
    ):
        pred_masks = []
        for i, pred_embedding in enumerate(pred_embeddings):
            sparse_embeddings, dense_embeddings = (
                self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embedding.unsqueeze(1),
                )
            )
            sparse_embeddings = sparse_embeddings.to(pred_embedding.dtype)
            low_res_masks, _ = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            orig_size = label_list[i].shape if not infer else label_list[i]
            # During inference, we have original size list in place of label list
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=orig_size,
            )
            pred_masks.append(pred_mask[:, 0])
        return pred_masks

    def _generate_and_postprocess_masks_multi(
        self,
        pred_embeddings,
        image_embeddings,
        resize_list,
        label_list,
        image_offset,
        infer=False,
    ):
        all_pred_masks = []
        for image_idx in range(len(pred_embeddings)):
            pred_masks = []
            for i, pred_embedding in enumerate(pred_embeddings[image_idx]):
                if pred_embedding is None:
                    pred_masks.append(None)
                    continue
                sparse_embeddings, dense_embeddings = (
                    self.model.visual_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=pred_embedding.unsqueeze(1),
                    )
                )
                sparse_embeddings = sparse_embeddings.to(pred_embedding.dtype)
                low_res_masks, _ = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[
                        image_offset[i] + image_idx
                    ].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                orig_size = (
                    label_list[image_idx][i].shape
                    if not infer
                    else label_list[image_idx][i]
                )
                # During inference, we have original size list in place of label list
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[image_idx][i],
                    original_size=orig_size,
                )
                pred_masks.append(pred_mask[:, 0])
            all_pred_masks.append(pred_masks)
        return tuple(all_pred_masks)

    def _calculate_losses(self, pred_masks, masks_list, output, image_paths):
        if len(image_paths[0]) > 1:
            return self._compute_loss_components_multi(pred_masks, masks_list, output)
        loss_components = self._compute_loss_components(pred_masks, masks_list, output)
        return loss_components

    def _compute_loss_components(self, pred_masks, masks_list, output):
        # Initialize loss components
        ce_loss = output.loss * self.ce_loss_weight
        mask_bce_loss = torch.tensor(0.0, device=ce_loss.device)
        mask_dice_loss = torch.tensor(0.0, device=ce_loss.device)
        num_masks = 0

        # Iterate over batch and compute mask-related losses
        for batch_idx, pred_mask in enumerate(pred_masks):
            if pred_mask.numel() > 0:  # Ensure pred_mask is not empty
                gt_mask = masks_list[batch_idx]
                # Resize gt_mask to match pred_mask if needed
                if gt_mask.shape[0] != pred_mask.shape[0]:
                    gt_mask = gt_mask[: pred_mask.shape[0]]

                assert (
                    gt_mask.shape[0] == pred_mask.shape[0]
                ), f"Shape mismatch: gt_mask {gt_mask.shape}, pred_mask {pred_mask.shape}"

                # Compute Binary Cross-Entropy Loss
                mask_bce_loss += (
                    compute_sigmoid_cross_entropy(
                        pred_mask, gt_mask, mask_count=gt_mask.shape[0]
                    )
                    * gt_mask.shape[0]
                )
                # Compute Dice Loss
                mask_dice_loss += (
                    calculate_dice_loss(pred_mask, gt_mask, mask_count=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
                num_masks += gt_mask.shape[0]

        # Normalize the losses
        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        # Aggregate all loss components
        total_loss = ce_loss + mask_loss
        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    def _compute_loss_components_multi(self, pred_masks, masks_list, output):
        # Initialize loss components
        ce_loss = output.loss * self.ce_loss_weight
        image_losses = torch.tensor(0.0, device=ce_loss.device)

        # Iterate over batch and compute mask-related losses
        for img_idx in range(len(pred_masks)):
            num_masks = 0
            mask_bce_loss = torch.tensor(0.0, device=ce_loss.device)
            mask_dice_loss = torch.tensor(0.0, device=ce_loss.device)
            for batch_idx, pred_mask in enumerate(pred_masks[img_idx]):
                if pred_mask is None:
                    assert (
                        masks_list[img_idx][batch_idx] is None
                    ), "no prediction but ground truth mask available"
                    continue
                if pred_mask.numel() > 0:  # Ensure pred_mask is not empty
                    gt_mask = masks_list[img_idx][batch_idx].to(pred_mask.device)
                    # Resize gt_mask to match pred_mask if needed
                    if gt_mask.shape[0] != pred_mask.shape[0]:
                        gt_mask = gt_mask[: pred_mask.shape[0]]

                    assert (
                        gt_mask.shape[0] == pred_mask.shape[0]
                    ), f"Shape mismatch: gt_mask {gt_mask.shape}, pred_mask {pred_mask.shape}"

                    # Compute Binary Cross-Entropy Loss
                    mask_bce_loss += (
                        compute_sigmoid_cross_entropy(
                            pred_mask, gt_mask, mask_count=gt_mask.shape[0]
                        )
                        * gt_mask.shape[0]
                    )
                    # Compute Dice Loss
                    mask_dice_loss += (
                        calculate_dice_loss(
                            pred_mask, gt_mask, mask_count=gt_mask.shape[0]
                        )
                        * gt_mask.shape[0]
                    )
                    num_masks += gt_mask.shape[0]
            # Normalize the losses
            mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
            mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)

            image_losses += mask_bce_loss + mask_dice_loss

        mask_loss = image_losses / len(pred_masks)

        # Aggregate all loss components
        total_loss = ce_loss + mask_loss
        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    def evaluate(
        self,
        global_enc_images,
        grounding_enc_images,
        input_ids,
        resize_list,
        orig_sizes,
        image_offset,
        max_new_tokens=None,
        bboxes=None,
        verbose=False,
        **kwargs,
    ):
        if max_new_tokens is None:
            raise ValueError(f"max_new_tokens not set")
        if global_enc_images is not None:
            image_paths = kwargs.get("image_paths")
            correspondence_feats = self.get_model().get_dino_features(image_paths)
            global_enc_images = (global_enc_images, correspondence_feats, image_offset)
        with torch.no_grad():
            generation_outputs = self.generate(
                images=global_enc_images,
                input_ids=input_ids,
                bboxes=bboxes,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

            output_hidden_states = generation_outputs.hidden_states
            generated_output_ids = generation_outputs.sequences

            if verbose:
                tokenizer = kwargs.pop("tokenizer", None)
                if tokenizer is not None:
                    output_ids = generated_output_ids[0][
                        generated_output_ids[0] != IMAGE_TOKEN_INDEX
                    ]
                    decoded = tokenizer.decode(output_ids, skip_special_tokens=False)
                    print(f"{decoded = }")

            seg_token_masks = self._create_seg_token_mask_multi(
                generated_output_ids, infer=True
            )
            if all(mask.count_nonzero() == 0 for mask in seg_token_masks):
                seg_token_mask = generated_output_ids[:, 1:] == self.seg_token_idx
                if seg_token_mask.count_nonzero() == 0:
                    return generated_output_ids, None
                # Adjusting for IMAGE_TOKEN_INDEX (assuming single image at start)
                seg_token_mask = torch.cat(
                    [
                        torch.zeros(
                            (seg_token_mask.shape[0], (self.img_emb_len - 1)),
                            dtype=torch.bool,
                            device=generated_output_ids.device,
                        ),
                        seg_token_mask,
                    ],
                    dim=1,
                )
                hidden_states, predicted_embeddings = self._process_hidden_states(
                    output_hidden_states, seg_token_mask, None, infer=True
                )
                image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
                # Generate and post-process masks
                pred_masks = self._generate_and_postprocess_masks(
                    predicted_embeddings,
                    image_embeddings,
                    resize_list,
                    orig_sizes,
                    infer=True,
                )
                return generated_output_ids, pred_masks

            # Process hidden states
            hidden_states, predicted_embeddings = self._process_hidden_states_multi(
                output_hidden_states, seg_token_masks, None, infer=True
            )
            image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
            # Generate and post-process masks
            pred_masks = self._generate_and_postprocess_masks_multi(
                predicted_embeddings,
                image_embeddings,
                resize_list,
                orig_sizes,
                image_offset,
                infer=True,
            )
        return generated_output_ids, pred_masks

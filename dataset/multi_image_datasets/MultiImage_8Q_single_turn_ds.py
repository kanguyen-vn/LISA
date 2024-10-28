import os
import re
import cv2
import glob
import json
import pickle
import random
from collections import Counter, defaultdict
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from dataset.utils.coco import COCO
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from dataset.utils.utils import (
    # MULTI_IMAGE_OBJECT_QUESTIONS,
    # COMMON_OBJECT_ANSWER_LIST,
    load_json_file,
    init_ade20k_part234,
    init_paco_lvis,
    init_pascal_part,
    _format,
    _write_list,
    _add_article,
    segm_to_mask,
)
from dataset.utils.blip_processors import (
    Blip2ImageTrainProcessor,
    Blip2ImageEvalProcessor,
)
from utils.utils import DEFAULT_IMAGE_TOKEN

AREA_THRESHOLD = None


class MultiImage8QSingleTurnDataset(torch.utils.data.Dataset):
    CLASSES = ("object",)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    ADE20K = "ade20k_part234"
    PACO_LVIS = "paco_lvis"
    PASCAL_PART = "pascal_part"

    def __init__(
        self,
        dataset_dir,
        tokenizer,
        global_image_encoder,
        epoch_samples=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        cosegm_data="ade20k_part234||paco_lvis||pascal_part",
        validation=False,
        random_sampling=True,
        multi_image_filepath_prefix="sample_8q_single_turn",
        mode=None,
        use_qformer=True,
    ):
        self.epoch_samples = epoch_samples
        self.num_classes_per_sample = num_classes_per_sample

        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.use_qformer = use_qformer
        if use_qformer:
            self.global_enc_processor = (
                Blip2ImageTrainProcessor()
                if mode == "train"
                else Blip2ImageEvalProcessor()
            )
        else:
            self.global_enc_processor = CLIPImageProcessor.from_pretrained(
                global_image_encoder
            )

        self.random_sampling = random_sampling

        self.data2api = {}
        self.dataset_dir = os.path.join(dataset_dir, "Part_Segm")
        self.coseg_ds_list = cosegm_data.split("||")
        self.mode = mode
        self.validation = self.mode in ["val", "test"]
        if mode is None:
            mode = "train" if not validation else "val"
        for ds in self.coseg_ds_list:
            classes, images, api = eval("init_{}".format(ds))(
                self.dataset_dir,
                mode if ds != self.ADE20K or mode != "test" else "val",
                AREA_THRESHOLD,
            )
            self.data2api[ds] = api
            print(
                f"\033[92m----COSEG-{mode.title()}: Loaded ObjectComparison - {ds} dataset ----\033[0m"
            )

        if multi_image_filepath_prefix.endswith("_1k"):
            self.multi_image_filepath = os.path.join(
                self.dataset_dir,
                f"{multi_image_filepath_prefix[:-3]}_{mode}_1k.json",
            )
        else:
            self.multi_image_filepath = os.path.join(
                self.dataset_dir,
                f"{multi_image_filepath_prefix}_{mode}.json",
            )
        # self.max_masks_per_class = max_masks_per_class
        self.json_data = load_json_file(self.multi_image_filepath)
        self.data_infos = self._init_data()

    def _init_data(self):
        self.json_data = [
            data
            for data in self.json_data
            if data["dataset_name"] in self.coseg_ds_list
        ]
        return self.json_data

    def __len__(self):
        return len(self.data_infos)

    def _set_len(self, length):
        self.epoch_samples = length

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def global_enc_process(self, image_path) -> torch.Tensor:
        if self.use_qformer:
            image = Image.open(image_path).convert("RGB")
            return self.global_enc_processor(image)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.global_enc_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

    def create_conversations(self, qa_pairs, num_images=2):
        selected_labels = []

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        questions = []
        begin_str = "The "
        img_token_list = []
        for i in range(num_images):
            img_token_list.append(f"{DEFAULT_IMAGE_TOKEN} (IMAGE{i+1})")
        begin_str = (
            begin_str
            + _write_list(img_token_list)
            + (
                " provide an overview of the pictures.\n"
                if num_images > 1
                else " provides an overview of the picture.\n"
            )
        )
        for i, qa_pair in enumerate(qa_pairs):
            question = qa_pair["q"]
            answer = qa_pair["a"]
            questions.append(question)
            if i == 0:
                question = begin_str + question
            conv.append_message(conv.roles[0], question)
            if self.mode == "test" and i == 0:
                conv.append_message(conv.roles[1], "")
                break
            conv.append_message(conv.roles[1], answer)
        conversations.append(conv.get_prompt())
        return questions, conversations, selected_labels

    def __getitem__(self, idx):
        data = self.data_infos[idx]
        dataset_name = data["dataset_name"]
        img_ids = data["img_ids"].copy()
        if dataset_name == self.PASCAL_PART or dataset_name == self.PACO_LVIS:
            img_ids = [int(img_id) for img_id in img_ids]
        img_id_idxs = list(range(len(img_ids)))

        if self.mode == "train":
            pass
            # random.shuffle(qa_pairs)
            # random.shuffle(img_id_idxs)
            # shuffled_img_ids = [img_ids[i] for i in img_id_idxs]
            # new_to_old_id = {new: old for new, old in zip(img_id_idxs, img_ids)}

        if dataset_name in [self.PACO_LVIS, self.ADE20K, self.PASCAL_PART]:
            coco_api = self.data2api[dataset_name]
            img_infos = coco_api.loadImgs(img_ids)
            filenames, heights, widths = [], [], []
            for img_info in img_infos:
                filenames.append(img_info["file_name"])
                heights.append(img_info["height"])
                widths.append(img_info["width"])
            if dataset_name == self.PACO_LVIS:
                image_paths = [
                    os.path.join(
                        self.dataset_dir.replace("Part_Segm/", "").replace(
                            "Part_Segm", ""
                        ),
                        "coco_2017",
                        fn,
                    )
                    for fn in filenames
                ]
            elif dataset_name == self.PASCAL_PART:
                image_paths = [
                    os.path.join(
                        self.dataset_dir,
                        "PASCAL-Part",
                        "VOCdevkit",
                        "VOC2010",
                        "JPEGImages",
                        fn,
                    )
                    for fn in filenames
                ]
            elif dataset_name == self.ADE20K:
                mode = "validation" if self.mode != "train" else "training"
                image_paths = [
                    os.path.join(
                        self.dataset_dir,
                        "ADE20KPart234",
                        "images",
                        mode,
                        fn,
                    )
                    for fn in filenames
                ]
                label_paths = [
                    image_path.replace(".jpg", ".png").replace(
                        "images", "annotations_detectron2_part"
                    )
                    for image_path in image_paths
                ]
                part_labels = [
                    np.array(Image.open(label_path)) for label_path in label_paths
                ]
        qa_pairs = []
        masks = [[] for _ in range(len(img_ids))]

        # IN EACH QA_PAIR
        imgwise_grounding_idxs = [0] * len(img_ids)
        qa_masks = [[] for _ in range(len(img_ids))]

        cur_answer = data["a"]
        re_pattern = r"\b(\w+)_\d{3}\b|\b(\w+)_\d{5}\b"
        cur_question = re.sub(re_pattern, lambda m: m.group(1) or m.group(2), data["q"])
        new_answer = ""

        occured_objects = []
        occured_parts = []
        match = re.search(re_pattern, cur_answer)
        while match:
            matched_string = match.group()
            st_idx = match.start()
            en_idx = match.end()
            matched_string_splitted = matched_string.split("_")
            if len(matched_string_splitted) > 2:
                cat_name = "_".join(matched_string_splitted[:-1])
                cat_code = matched_string_splitted[-1]
            else:
                cat_name, cat_code = matched_string_splitted
            cat_type = "object" if len(cat_code) == 3 else "part"
            img = cat_code[0]

            if cat_type == "object":
                if matched_string in occured_objects:
                    img_tokens = f"(IMAGE{img})"
                else:
                    occured_objects.append(matched_string)
                    img_tokens = f"[SEG] (IMAGE{img})"
                    grounding = data["groundings"][f"img_{img}"][
                        imgwise_grounding_idxs[int(img) - 1]
                    ]
                    segm = grounding["segmentation"]
                    cur_mask = segm_to_mask(
                        segm, heights[int(img) - 1], widths[int(img) - 1]
                    )
                    imgwise_grounding_idxs[int(img) - 1] += 1
                    qa_masks[int(img) - 1].append(cur_mask)
                grounded_phrase = f"{grounding['name']}_{int(cat_code[1:])}"

            else:
                if matched_string in occured_parts:
                    img_tokens = f"(IMAGE{img})"
                else:
                    occured_parts.append(matched_string)
                    img_tokens = f"[SEG] (IMAGE{img})"
                    grounding = data["groundings"][f"img_{img}"][
                        imgwise_grounding_idxs[int(img) - 1]
                    ]
                    cur_mask = segm_to_mask(
                        grounding["segmentation"],
                        heights[int(img) - 1],
                        widths[int(img) - 1],
                    )
                    imgwise_grounding_idxs[int(img) - 1] += 1
                    obj_ann_id = int(grounding["obj_ann_id"])
                    obj_ann = coco_api.loadAnns(obj_ann_id)[0]
                    qa_masks[int(img) - 1].append(cur_mask)
                obj_cat_id = obj_ann["category_id"]
                obj_cat = coco_api.loadCats(obj_cat_id)[0]["name"]
                grounded_phrase = f"{obj_cat}_{int(cat_code[1:3])}'s {grounding['name']}_{int(cat_code[3:])}"
            new_answer += (
                cur_answer[:st_idx] + f"<p> {grounded_phrase} </p> {img_tokens})"
            )
            cur_answer = cur_answer[en_idx:]
            match = re.search(re_pattern, cur_answer)
        new_answer += cur_answer
        qa_pairs.append({"q": cur_question, "a": new_answer})
        for i in range(len(img_ids)):
            masks[i].extend(qa_masks[i])
        # QA PAIR LOOP END

        # Load and process the image
        images = [cv2.imread(image_path) for image_path in image_paths]
        images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        global_enc_imgs = [
            self.global_enc_process(image_path) for image_path in image_paths
        ]
        images = [self.transform.apply_image(image) for image in images]
        image_resizes = [image.shape[:2] for image in images]
        grounding_enc_imgs = [
            self.grounding_enc_processor(
                torch.from_numpy(image).permute(2, 0, 1).contiguous()
            )
            for image in images
        ]

        # Generate questions and answers
        questions, conversations, selected_labels = self.create_conversations(
            qa_pairs, len(img_ids)
        )

        # new_masks = []
        new_masks = [None]
        image_resizes = [None, *image_resizes]
        for i, cur_masks in enumerate(masks):
            if len(cur_masks) == 0:
                cur_masks = None
            else:
                cur_masks = np.stack(cur_masks)
                cur_masks = torch.from_numpy(cur_masks)
            new_masks.append(cur_masks)

        masks = new_masks
        # masks = [torch.from_numpy(np.stack(cur_masks)) for cur_masks in masks]
        label = [
            (
                torch.ones(cur_masks.shape[1], cur_masks.shape[2]) * self.IGNORE_LABEL
                if cur_masks is not None
                else None
            )
            for cur_masks in masks
        ]

        assert len(conversations) == 1
        if self.mode != "test":
            assert conversations[0].count("[SEG]") == sum(
                [(mask.shape[0] if mask is not None else 0) for mask in masks]
            )

        # assert conversations[0].count("<image>") == 2, f"{conversations = }"
        # set bboxes to None for segmentation datasets
        bboxes = None

        global_enc_img = torch.stack(global_enc_imgs, dim=0)
        grounding_enc_img = torch.stack(grounding_enc_imgs, dim=0)
        # print(len(qa_pairs), img_ids, masks[0].shape)
        return (
            image_paths,
            global_enc_img,
            grounding_enc_img,
            bboxes,
            conversations,
            masks,
            label,
            image_resizes,
            questions,
            selected_labels,
        )

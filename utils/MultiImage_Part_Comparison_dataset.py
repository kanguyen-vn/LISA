import os
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
from .coco import COCO
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from .multi_image_utils import (
    MULTI_IMAGE_COMMON_PART_QUESTIONS,
    MULTI_IMAGE_UNIQUE_PART_QUESTIONS,
    MULTI_IMAGE_COMMON_PART_ANSWER_LIST,
    MULTI_IMAGE_UNIQUE_PART_ANSWER_LIST,
    ID_OBJECT_ANSWER_LIST,
    load_json_file,
    init_ade20k_part234,
    init_paco_lvis,
    init_part_image_net,
    _format,
    _write_list,
    _add_article,
)
from model.llava.constants import DEFAULT_IMAGE_TOKEN

AREA_THRESHOLD = 0.02


class MultiImagePartComparisonDataset(torch.utils.data.Dataset):
    CLASSES = ("object",)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    ADE20K = "ade20k_part234"
    PACO_LVIS = "paco_lvis"
    PART_IMAGE_NET = "part_image_net"

    def __init__(
        self,
        dataset_dir,
        tokenizer,
        global_image_encoder,
        epoch_samples=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        cosegm_data="ade20k_part234||paco_lvis||part_image_net",
        validation=False,
        random_sampling=True,
        multi_image_filepath_prefix="",
        dino_feats_path="",
        # max_masks_per_class=3,
        mode=None,
    ):
        self.epoch_samples = epoch_samples
        self.num_classes_per_sample = num_classes_per_sample

        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(
            global_image_encoder
        )

        self.validation = validation
        self.random_sampling = random_sampling

        self.data2imgs = {}
        self.data2api = {}
        self.data2classes = {}
        self.dataset_dir = os.path.join(dataset_dir, "Part_Segm")
        self.coseg_ds_list = cosegm_data.split("||")
        self.mode = mode
        self.validation = self.mode in ["val", "test"]
        self.begin_str = f"{DEFAULT_IMAGE_TOKEN} {DEFAULT_IMAGE_TOKEN}\n"
        if mode is None:
            mode = "train" if not self.validation else "val"
        for ds in self.coseg_ds_list:
            classes, images, labels = eval("init_{}".format(ds))(
                self.dataset_dir, mode if ds != self.ADE20K or mode != "test" else "val"
            )
            self.data2imgs[ds] = images
            self.data2api[ds] = labels
            self.data2classes[ds] = classes
            print(
                f"\033[92m----COSEG-{mode.title()}: Loaded PartComparison - {ds} dataset ----\033[0m"
            )

        self.multi_image_filepath = os.path.join(
            self.dataset_dir,
            f"{multi_image_filepath_prefix}_{mode}.json",
        )
        self.dino_feats_path = dino_feats_path
        self.json_data = load_json_file(self.multi_image_filepath)
        self.data_infos = self._init_data()

    def _init_data(self):
        self.json_data = [
            data
            for data in self.json_data
            if data["dataset_name"] in self.coseg_ds_list and not data["is_object"]
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

    def create_conversations(self, sampled_anns):
        questions = []
        answers = []
        selected_labels = []

        for i, pair in enumerate(sampled_anns):
            obj_cat1, obj_cat2 = pair["cats"]
            obj_cat1 = obj_cat1.lower().strip()
            obj_cat2 = obj_cat2.lower().strip()

            if obj_cat1 == obj_cat2:
                a_class_name = _add_article(obj_cat1, "a")
                answer_id = random.choice(ID_OBJECT_ANSWER_LIST["same"])
                answer_id = answer_id.format(
                    a_class_name=f"<p> {a_class_name} </p> [SEG] (IMAGE1) [SEG] (IMAGE2)"
                )
                selected_labels.extend([a_class_name, a_class_name])
            else:
                all_class_names = [
                    _add_article(obj_cat1, "a"),
                    _add_article(obj_cat2, "a"),
                ]
                selected_labels.extend(all_class_names)
                all_class_names = [
                    f"<p> {a_class_name} </p> [SEG] (IMAGE{idx + 1})"
                    for idx, a_class_name in enumerate(all_class_names)
                ]

                all_class_names = _write_list(all_class_names)
                answer_id = random.choice(ID_OBJECT_ANSWER_LIST["different"])
                answer_id = answer_id.format(a_class_names=all_class_names)

            if "unique_first" not in pair or not pair["unique_first"]:
                if pair["common"] is not None:
                    commons = pair["common"]
                    question_templates = MULTI_IMAGE_COMMON_PART_QUESTIONS
                    answer_template = (
                        random.choice(MULTI_IMAGE_COMMON_PART_ANSWER_LIST["singular"])
                        if len(commons) == 1
                        else random.choice(
                            MULTI_IMAGE_COMMON_PART_ANSWER_LIST["plural"]
                        )
                    )
                    the_phrases, a_phrases = [], []

                    for common in commons:
                        part_cat = common["part_cat"]
                        masks1, masks2 = common["masks"]
                        img1_tokens = "[SEG] (IMAGE1)"
                        img2_tokens = "[SEG] (IMAGE2)"
                        for article, phrase_list in zip(
                            ["the", "a"], [the_phrases, a_phrases]
                        ):
                            grounded_phrase = _add_article(part_cat, article)
                            phrase = f"<p> {grounded_phrase} </p> {img1_tokens} {img2_tokens}"
                            phrase_list.append(phrase)
                        selected_labels.extend([part_cat] * 2)

                    the_phrases = _write_list(the_phrases)
                    a_phrases = _write_list(a_phrases)
                    answer = _format(
                        answer_template, the_parts=the_phrases, a_parts=a_phrases
                    )
                    answer = f"{answer_id} {answer}"

                    question_template = random.choice(question_templates)
                    questions.append(question_template)
                    answers.append(answer)

                if pair["unique"] is not None:
                    unique = pair["unique"]
                    question_templates = MULTI_IMAGE_UNIQUE_PART_QUESTIONS
                    answer_template = (
                        random.choice(MULTI_IMAGE_UNIQUE_PART_ANSWER_LIST["singular"])
                        if sum(len(x) for x in unique["part_cats"]) == 1
                        else random.choice(
                            MULTI_IMAGE_UNIQUE_PART_ANSWER_LIST["plural"]
                        )
                    )
                    the_phrases, a_phrases = [], []
                    for i, (part_cats, masks) in enumerate(
                        zip(unique["part_cats"], unique["masks"])
                    ):
                        for part_cat, mask in zip(part_cats, masks):
                            for article, phrase_list in zip(
                                ["the", "a"], [the_phrases, a_phrases]
                            ):
                                grounded_phrase = _add_article(part_cat, article)
                                phrase_list.append(
                                    f"<p> {grounded_phrase} </p> [SEG] (IMAGE{i + 1})"
                                )
                            selected_labels.append(part_cat)
                    the_phrases = _write_list(the_phrases)
                    a_phrases = _write_list(a_phrases)
                    answer = _format(
                        answer_template, the_parts=the_phrases, a_parts=a_phrases
                    )
                    answer = f"{answer_id} {answer}"

                    question_template = random.choice(question_templates)
                    questions.append(question_template)
                    answers.append(answer)
            else:
                if pair["unique"] is not None:
                    unique = pair["unique"]
                    question_templates = MULTI_IMAGE_UNIQUE_PART_QUESTIONS
                    answer_template = (
                        random.choice(MULTI_IMAGE_UNIQUE_PART_ANSWER_LIST["singular"])
                        if sum(len(x) for x in unique["part_cats"]) == 1
                        else random.choice(
                            MULTI_IMAGE_UNIQUE_PART_ANSWER_LIST["plural"]
                        )
                    )
                    the_phrases, a_phrases = [], []
                    for i, (part_cats, masks) in enumerate(
                        zip(unique["part_cats"], unique["masks"])
                    ):
                        for part_cat, mask in zip(part_cats, masks):
                            for article, phrase_list in zip(
                                ["the", "a"], [the_phrases, a_phrases]
                            ):
                                grounded_phrase = _add_article(part_cat, article)
                                phrase_list.append(
                                    f"<p> {grounded_phrase} </p> [SEG] (IMAGE{i + 1})"
                                )
                            selected_labels.append(part_cat)
                    the_phrases = _write_list(the_phrases)
                    a_phrases = _write_list(a_phrases)
                    answer = _format(
                        answer_template, the_parts=the_phrases, a_parts=a_phrases
                    )
                    answer = f"{answer_id} {answer}"

                    question_template = random.choice(question_templates)
                    questions.append(question_template)
                    answers.append(answer)
                if pair["common"] is not None:
                    commons = pair["common"]
                    question_templates = MULTI_IMAGE_COMMON_PART_QUESTIONS
                    answer_template = (
                        random.choice(MULTI_IMAGE_COMMON_PART_ANSWER_LIST["singular"])
                        if len(commons) == 1
                        else random.choice(
                            MULTI_IMAGE_COMMON_PART_ANSWER_LIST["plural"]
                        )
                    )
                    the_phrases, a_phrases = [], []

                    for common in commons:
                        part_cat = common["part_cat"]
                        masks1, masks2 = common["masks"]
                        img1_tokens = "[SEG] (IMAGE1)"
                        img2_tokens = "[SEG] (IMAGE2)"
                        for article, phrase_list in zip(
                            ["the", "a"], [the_phrases, a_phrases]
                        ):
                            grounded_phrase = _add_article(part_cat, article)
                            phrase = f"<p> {grounded_phrase} </p> {img1_tokens} {img2_tokens}"
                            phrase_list.append(phrase)
                        selected_labels.extend([part_cat] * 2)

                    the_phrases = _write_list(the_phrases)
                    a_phrases = _write_list(a_phrases)
                    answer = _format(
                        answer_template, the_parts=the_phrases, a_parts=a_phrases
                    )
                    answer = f"{answer_id} {answer}"

                    question_template = random.choice(question_templates)
                    questions.append(question_template)
                    answers.append(answer)

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if i == 0:
                question = self.begin_str + question
            conv.append_message(conv.roles[0], question)
            if self.mode == "test" and i == 0:
                conv.append_message(conv.roles[1], "")
                break
            conv.append_message(conv.roles[1], answer)
        conversations.append(conv.get_prompt())
        return questions, conversations, selected_labels

    def __getitem__(self, idx):
        data = self.data_infos[idx].copy()
        dataset_name = data["dataset_name"]
        img_id1, img_id2 = data["img_ids"]
        pairs = data["pairs"]

        pairs = [
            pair
            for pair in pairs
            if not (pair["common"] is None and pair["unique"] is None)
        ]

        if self.mode == "train":
            flip_images = random.choice([True, False])
            if flip_images:
                img_id1, img_id2 = img_id2, img_id1
                new_pairs = []
                for pair in pairs:
                    new_pair = {
                        k: v[::-1]
                        for k, v in pair.items()
                        if isinstance(v, tuple) or isinstance(v, list)
                    }
                    new_pair["common"] = new_pair["unique"] = None
                    if pair["common"] is not None:
                        new_commons = []
                        for common in pair["common"]:
                            new_common = {
                                k: (
                                    v[::-1]
                                    if isinstance(v, tuple) or isinstance(v, list)
                                    else v
                                )
                                for k, v in common.items()
                            }
                            new_commons.append(new_common)
                        new_pair["common"] = new_commons
                    if pair["unique"] is not None:
                        new_unique = {
                            k: (
                                v[::-1]
                                if isinstance(v, tuple) or isinstance(v, list)
                                else v
                            )
                            for k, v in pair["unique"].items()
                        }
                        new_pair["unique"] = new_unique
                    new_pairs.append(new_pair)
                pairs = new_pairs

        if dataset_name in [self.PACO_LVIS]:
            class_map = self.data2classes[dataset_name]
            coco_api = self.data2api[dataset_name]
            img_info1, img_info2 = coco_api.loadImgs([img_id1, img_id2])
            file_name1, file_name2 = img_info1["file_name"], img_info2["file_name"]
            mode = None
            image_path1 = os.path.join(
                self.dataset_dir.replace("Part_Segm/", "").replace("Part_Segm", ""),
                "coco_2017",
                file_name1,
            )
            image_path2 = os.path.join(
                self.dataset_dir.replace("Part_Segm/", "").replace("Part_Segm", ""),
                "coco_2017",
                file_name2,
            )
        elif dataset_name in [self.ADE20K]:
            class_map = self.data2classes[dataset_name]
            coco_api = self.data2api[dataset_name]
            img_info1, img_info2 = coco_api.loadImgs([img_id1, img_id2])
            file_name1, file_name2 = img_info1["file_name"], img_info2["file_name"]
            mode = "validation" if self.mode != "train" else "training"
            image_path1 = os.path.join(
                self.dataset_dir, "ADE20KPart234", "images", mode, file_name1
            )
            image_path2 = os.path.join(
                self.dataset_dir, "ADE20KPart234", "images", mode, file_name2
            )
            label_path1 = image_path1.replace(".jpg", ".png").replace(
                "images", "annotations_detectron2_part"
            )
            label1 = np.array(Image.open(label_path1))
            label_path2 = image_path2.replace(".jpg", ".png").replace(
                "images", "annotations_detectron2_part"
            )
            label2 = np.array(Image.open(label_path2))

        elif dataset_name in [self.PART_IMAGE_NET]:
            class_map = self.data2classes[dataset_name]
            (coco_api, coco_api_part) = self.data2api[dataset_name]
            img_info1, img_info2 = coco_api.loadImgs([img_id1, img_id2])
            file_name1, file_name2 = img_info1["file_name"], img_info2["file_name"]
            mode = self.mode
            image_path1 = os.path.join(
                self.dataset_dir, "PartImageNet", "images", mode, file_name1
            )
            image_path2 = os.path.join(
                self.dataset_dir, "PartImageNet", "images", mode, file_name2
            )
        else:
            raise NotImplementedError(f"unsupported dataset {dataset_name}")

        pairs = sorted(pairs, key=lambda x: sum(x["visibility"]))
        if self.mode == "train":
            sampled_pairs = (
                pairs[: self.num_classes_per_sample]
                if len(pairs) > self.num_classes_per_sample
                else pairs
            )
        else:
            sampled_pairs = [pairs[0]]
        sampled_anns = []
        for pair in sampled_pairs:
            obj_pair_ann_ids = pair["ann_ids"]
            obj_pair_anns = [coco_api.loadAnns(ann_ids) for ann_ids in obj_pair_ann_ids]
            obj_pair_masks = [
                [coco_api.annToMask(ann) for ann in anns] for anns in obj_pair_anns
            ]
            obj_cats = pair["cats"]
            obj_mask1, obj_mask2 = [
                sum(masks, np.zeros_like(masks[0])) for masks in obj_pair_masks
            ]
            new_pair = {"cats": obj_cats, "obj_masks": (obj_mask1, obj_mask2)}

            new_pair["common"] = new_pair["unique"] = None
            if pair["common"] is not None:
                new_commons = []
                for common in sorted(
                    pair["common"], key=lambda x: sum(x["part_visibility"])
                ):
                    part_cat = common["part_cat"]

                    if dataset_name in [self.PACO_LVIS, self.PART_IMAGE_NET]:
                        api_part = (
                            coco_api
                            if dataset_name == self.PACO_LVIS
                            else coco_api_part
                        )
                        paired_anns = [
                            api_part.loadAnns(ann_ids)
                            for ann_ids in common["part_ann_ids"]
                        ]
                        paired_masks = [
                            [api_part.annToMask(ann) for ann in anns]
                            for anns in paired_anns
                        ]
                        mask1, mask2 = [
                            sum(masks, np.zeros_like(masks[0]))
                            for masks in paired_masks
                        ]
                    elif dataset_name == self.ADE20K:
                        part_cat_ids = common["part_cat_ids"]

                        masks1 = [
                            ((obj_mask1 == 1) & (label1 == part_cat_id))
                            for part_cat_id in part_cat_ids[0]
                        ]
                        masks2 = [
                            ((obj_mask2 == 1) & (label2 == part_cat_id))
                            for part_cat_id in part_cat_ids[1]
                        ]
                        mask1, mask2 = [
                            sum(masks, np.zeros_like(masks[0]))
                            for masks in (masks1, masks2)
                        ]

                    mask1 = np.where(mask1 > 0, 1, 0)
                    mask2 = np.where(mask2 > 0, 1, 0)

                    new_commons.append({"part_cat": part_cat, "masks": (mask1, mask2)})
                new_pair["common"] = new_commons
            if pair["unique"] is not None:
                unique = pair["unique"]
                part_cats = unique["part_cats"]
                if dataset_name in [self.PACO_LVIS, self.PART_IMAGE_NET]:
                    api_part = (
                        coco_api if dataset_name == self.PACO_LVIS else coco_api_part
                    )
                    part_ann_ids1, part_ann_ids2 = unique["part_ann_ids"]

                    part_anns1 = [
                        api_part.loadAnns(p_ann_ids) for p_ann_ids in part_ann_ids1
                    ]
                    part_anns2 = [
                        api_part.loadAnns(p_ann_ids) for p_ann_ids in part_ann_ids2
                    ]

                    all_part_masks1 = [
                        [api_part.annToMask(ann) for ann in anns] for anns in part_anns1
                    ]
                    all_part_masks2 = [
                        [api_part.annToMask(ann) for ann in anns] for anns in part_anns2
                    ]
                elif dataset_name == self.ADE20K:
                    part_cat_ids1, part_cat_ids2 = unique["part_cat_ids"]
                    all_part_masks1 = [
                        [
                            ((obj_mask1 == 1) & (label1 == part_cat_id))
                            for part_cat_id in p_cat_ids
                        ]
                        for p_cat_ids in part_cat_ids1
                    ]
                    all_part_masks2 = [
                        [
                            ((obj_mask2 == 1) & (label2 == part_cat_id))
                            for part_cat_id in p_cat_ids
                        ]
                        for p_cat_ids in part_cat_ids2
                    ]

                part_masks1 = [
                    sum(masks, np.zeros_like(masks[0])) for masks in all_part_masks1
                ]
                part_masks2 = [
                    sum(masks, np.zeros_like(masks[0])) for masks in all_part_masks2
                ]
                part_masks1 = [
                    np.where(part_mask > 0, 1, 0) for part_mask in part_masks1
                ]
                part_masks2 = [
                    np.where(part_mask > 0, 1, 0) for part_mask in part_masks2
                ]
                part_masks = (part_masks1, part_masks2)

                new_pair["unique"] = {"part_cats": part_cats, "masks": part_masks}

            if pair["common"] is not None and pair["unique"] is not None:
                new_pair["unique_first"] = random.choice([True, False])

            sampled_anns.append(new_pair)

        # Load and process the image
        image1 = cv2.imread(image_path1)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        global_enc_img1 = self.global_enc_processor.preprocess(
            image1, return_tensors="pt"
        )["pixel_values"][0]
        image1 = self.transform.apply_image(image1)
        image_resize1 = image1.shape[:2]
        grounding_enc_img1 = self.grounding_enc_processor(
            torch.from_numpy(image1).permute(2, 0, 1).contiguous()
        )
        image2 = cv2.imread(image_path2)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        global_enc_img2 = self.global_enc_processor.preprocess(
            image2, return_tensors="pt"
        )["pixel_values"][0]
        image2 = self.transform.apply_image(image2)
        image_resize2 = image2.shape[:2]
        grounding_enc_img2 = self.grounding_enc_processor(
            torch.from_numpy(image2).permute(2, 0, 1).contiguous()
        )

        correspondence_feats1 = correspondence_feats2 = None

        if self.dino_feats_path != "":
            correspondence_feats_path1 = os.path.join(
                self.dino_feats_path,
                dataset_name,
                (f"{mode}/" if mode is not None else "") + f"{img_id1}.out",
            )
            correspondence_feats_path2 = os.path.join(
                self.dino_feats_path,
                dataset_name,
                (f"{mode}/" if mode is not None else "") + f"{img_id2}.out",
            )
            correspondence_feats1 = (
                torch.load(correspondence_feats_path1).squeeze(0).squeeze(0)
            )
            correspondence_feats2 = (
                torch.load(correspondence_feats_path2).squeeze(0).squeeze(0)
            )

        # Generate questions and answers
        questions, conversations, selected_labels = self.create_conversations(
            sampled_anns
        )

        masks1, masks2 = [], []
        for pair in sampled_anns:
            if "unique_first" not in pair or not pair["unique_first"]:
                # print("common first")
                if pair["common"] is not None:
                    masks1.append(pair["obj_masks"][0])
                    masks2.append(pair["obj_masks"][1])
                    commons = pair["common"]
                    # print(f"{len(commons) = }")
                    for common in commons:
                        # print(f"{len(common['masks']) = }")
                        masks1.append(common["masks"][0])
                        masks2.append(common["masks"][1])
                if pair["unique"] is not None:
                    masks1.append(pair["obj_masks"][0])
                    masks2.append(pair["obj_masks"][1])
                    unique = pair["unique"]
                    # print(f"{len(unique['masks'][0]) = }")
                    # print(f"{len(unique['masks'][1]) = }")
                    unique = pair["unique"]
                    masks1.extend(unique["masks"][0])
                    masks2.extend(unique["masks"][1])
            else:
                # print("unique first")
                if pair["unique"] is not None:
                    masks1.append(pair["obj_masks"][0])
                    masks2.append(pair["obj_masks"][1])
                    unique = pair["unique"]
                    # print(f"{len(unique['masks'][0]) = }")
                    # print(f"{len(unique['masks'][1]) = }")
                    unique = pair["unique"]
                    masks1.extend(unique["masks"][0])
                    masks2.extend(unique["masks"][1])
                if pair["common"] is not None:
                    masks1.append(pair["obj_masks"][0])
                    masks2.append(pair["obj_masks"][1])
                    commons = pair["common"]
                    # print(f"{len(commons) = }")
                    for common in commons:
                        # print(f"{len(common['masks']) = }")
                        masks1.append(common["masks"][0])
                        masks2.append(common["masks"][1])

        if masks1:
            masks1 = np.stack(masks1, axis=0)
            masks1 = torch.from_numpy(masks1)
        else:
            masks1 = None
        label1 = torch.ones(img_info1["height"], img_info1["width"]) * self.IGNORE_LABEL
        if masks2:
            masks2 = np.stack(masks2, axis=0)
            masks2 = torch.from_numpy(masks2)
        else:
            masks2 = None
        label2 = torch.ones(img_info2["height"], img_info2["width"]) * self.IGNORE_LABEL
        masks = (masks1, masks2)
        label = (label1, label2)

        assert len(conversations) == 1

        # print(f"part {conversations}\n")

        if self.mode != "test":
            assert conversations[0].count("[SEG]") == (
                masks1.shape[0] if masks1 is not None else 0
            ) + (
                masks2.shape[0] if masks2 is not None else 0
            ), f"{conversations = }\nmasks1 {masks1.shape if masks1 is not None else None}, masks2 {masks2.shape if masks2 is not None else None}"
        assert (
            conversations[0].count("<image>") == 2
        ), f"{conversations = }\n\n{sampled_anns = }\n\n{data = }\n\n{flip_images = }"
        # set bboxes to None for segmentation datasets
        bboxes = None

        image_paths = [image_path1, image_path2]
        global_enc_img = torch.stack([global_enc_img1, global_enc_img2], dim=0)
        grounding_enc_img = torch.stack([grounding_enc_img1, grounding_enc_img2], dim=0)
        image_resizes = [image_resize1, image_resize2]
        if correspondence_feats1 is not None:
            correspondence_feats = torch.stack(
                [correspondence_feats1, correspondence_feats2], dim=0
            )
            global_enc_img = (global_enc_img, correspondence_feats)

        return (
            image_paths,
            grounding_enc_img,
            global_enc_img,
            bboxes,
            conversations,
            masks,
            label,
            image_resizes,
            questions,
            selected_labels,
        )

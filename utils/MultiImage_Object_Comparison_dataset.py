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
    MULTI_IMAGE_OBJECT_QUESTIONS,
    COMMON_OBJECT_ANSWER_LIST,
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


class MultiImageObjectComparisonDataset(torch.utils.data.Dataset):
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

        self.random_sampling = random_sampling

        self.data2imgs = {}
        self.data2classes = {}
        self.data2api = {}
        self.dataset_dir = os.path.join(dataset_dir, "Part_Segm")
        self.coseg_ds_list = cosegm_data.split("||")
        self.mode = mode
        self.validation = self.mode in ["val", "test"]
        self.begin_str = f"{DEFAULT_IMAGE_TOKEN} {DEFAULT_IMAGE_TOKEN}\n"
        if mode is None:
            mode = "train" if not validation else "val"
        for ds in self.coseg_ds_list:
            classes, images, api = eval("init_{}".format(ds))(
                self.dataset_dir,
                mode if ds != self.ADE20K or mode != "test" else "val",
                AREA_THRESHOLD,
            )
            self.data2imgs[ds] = images
            self.data2classes[ds] = classes
            self.data2api[ds] = api
            print(
                f"\033[92m----COSEG-{mode.title()}: Loaded ObjectComparison - {ds} dataset ----\033[0m"
            )

        self.multi_image_filepath = os.path.join(
            self.dataset_dir,
            f"{multi_image_filepath_prefix}_{mode}.json",
        )
        # self.max_masks_per_class = max_masks_per_class
        self.dino_feats_path = dino_feats_path
        self.json_data = load_json_file(self.multi_image_filepath)
        self.data_infos = self._init_data()

    def _init_data(self):
        self.json_data = [
            data
            for data in self.json_data
            if data["dataset_name"] in self.coseg_ds_list and data["is_object"]
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

        question_templates = MULTI_IMAGE_OBJECT_QUESTIONS

        question_template = random.choice(question_templates)
        questions.append(question_template)

        if len(sampled_anns) == 1:
            answer_template = random.choice(COMMON_OBJECT_ANSWER_LIST["singular"])
        else:
            answer_template = random.choice(COMMON_OBJECT_ANSWER_LIST["plural"])

        the_phrases, a_phrases = [], []
        for ann in sampled_anns:
            cat = ann["cat"]
            masks1, masks2 = ann["masks"]
            img1_tokens = "[SEG] (IMAGE1)"
            img2_tokens = "[SEG] (IMAGE2)"
            for article, phrase_list in zip(["the", "a"], [the_phrases, a_phrases]):
                grounded_phrase = _add_article(cat, article)
                phrase = f"<p> {grounded_phrase} </p> {img1_tokens} {img2_tokens}"
                phrase_list.append(phrase)
            selected_labels.extend([cat] * (len(masks1) + len(masks2)))

        the_phrases = _write_list(the_phrases)
        a_phrases = _write_list(a_phrases)

        answers.append(
            _format(
                answer_template, the_class_names=the_phrases, a_class_names=a_phrases
            )
        )

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
        data = self.data_infos[idx]
        dataset_name = data["dataset_name"]
        img_id1, img_id2 = data["img_ids"]
        commons = data["common"]

        if self.mode == "train":
            flip_images = random.choice([True, False])
            if flip_images:
                img_id1, img_id2 = img_id2, img_id1
                new_commons = []
                for common in commons:
                    new_common = {
                        k: (
                            v[::-1]
                            if isinstance(v, tuple) or isinstance(v, list)
                            else v
                        )
                        for k, v in common.items()
                    }
                    new_commons.append(new_common)
                commons = new_commons

        if dataset_name in [self.PACO_LVIS, self.ADE20K]:
            class_map = self.data2classes[dataset_name]["object"]
            _, coco_api = self.data2list[dataset_name]
            img_info1, img_info2 = coco_api.loadImgs([img_id1, img_id2])
            file_name1, file_name2 = img_info1["file_name"], img_info2["file_name"]

            if dataset_name == self.PACO_LVIS:
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
            else:
                mode = "validation" if self.mode != "train" else "training"
                image_path1 = os.path.join(
                    self.dataset_dir, "ADE20KPart234", "images", mode, file_name1
                )
                image_path2 = os.path.join(
                    self.dataset_dir, "ADE20KPart234", "images", mode, file_name2
                )

        elif dataset_name in [self.PART_IMAGE_NET]:
            class_map = self.data2classes[dataset_name]["object"]
            _, (coco_api, coco_api_part) = self.data2list[dataset_name]
            img_info1, img_info2 = coco_api.loadImgs([img_id1, img_id2])
            file_name1, file_name2 = img_info1["file_name"], img_info2["file_name"]
            mode = self.mode
            image_path1 = os.path.join(
                self.dataset_dir, "PartImageNet", "images", mode, file_name1
            )
            image_path2 = os.path.join(
                self.dataset_dir, "PartImageNet", "images", mode, file_name2
            )

        sampled_anns = []
        commons = sorted(commons, key=lambda x: sum(x["visibility"]))
        for common in commons:
            cat = common["cat"]
            ann_ids = common["ann_ids"]
            anns1 = coco_api.loadAnns(ann_ids[0])
            anns2 = coco_api.loadAnns(ann_ids[1])

            masks_anns1 = [coco_api.annToMask(ann) for ann in anns1]
            masks_anns2 = [coco_api.annToMask(ann) for ann in anns2]

            combined_masks_anns1 = sum(masks_anns1, np.zeros_like(masks_anns1[0]))
            combined_masks_anns1 = np.where(combined_masks_anns1 > 0, 1, 0)
            combined_masks_anns2 = sum(masks_anns2, np.zeros_like(masks_anns2[0]))
            combined_masks_anns2 = np.where(combined_masks_anns2 > 0, 1, 0)

            sampled_anns.append(
                {"cat": cat, "masks": (combined_masks_anns1, combined_masks_anns2)}
            )

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
        for common in sampled_anns:
            # anns1, anns2 = common["anns"]
            # masks_per_cat1 = [coco_api.annToMask(ann) for ann in anns1]
            # masks_per_cat2 = [coco_api.annToMask(ann) for ann in anns2]
            masks_per_cat1, masks_per_cat2 = common["masks"]
            masks1.append(masks_per_cat1)
            masks2.append(masks_per_cat2)

        masks1 = np.stack(masks1, axis=0)
        masks1 = torch.from_numpy(masks1)
        label1 = torch.ones(masks1.shape[1], masks1.shape[2]) * self.IGNORE_LABEL
        masks2 = np.stack(masks2, axis=0)
        masks2 = torch.from_numpy(masks2)
        label2 = torch.ones(masks2.shape[1], masks2.shape[2]) * self.IGNORE_LABEL
        masks = (masks1, masks2)
        label = (label1, label2)

        assert len(conversations) == 1
        if self.mode != "test":
            assert conversations[0].count("[SEG]") == masks1.shape[0] + masks2.shape[0]

        assert conversations[0].count("<image>") == 2, f"{conversations = }"
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
            conversations,
            masks,
            label,
            image_resizes,
            questions,
            selected_labels,
        )

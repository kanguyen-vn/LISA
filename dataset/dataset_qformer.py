import numpy as np
import torch
from itertools import zip_longest

from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from dataset.segm_datasets.RefCOCO_Segm_ds import ReferSegmDataset
from dataset.multi_image_datasets.MultiImage_8Q_single_turn_ds import (
    MultiImage8QSingleTurnDataset,
)
from utils.utils import (
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
)


class HybridDatasetBase(torch.utils.data.Dataset):
    PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(
        self,
        dataset_dir,
        tokenizer,
        global_image_encoder,
        dataset,
        datasets_config,
        epoch_samples=500 * 8 * 2 * 10,
        batch_size=2,
        precision="fp32",
        image_size=224,
        num_classes_per_sample=3,
        sample_rate=None,
    ):
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.global_image_encoder = global_image_encoder
        self.dataset = dataset
        self.datasets_config = datasets_config
        self.epoch_samples = epoch_samples
        self.batch_size = batch_size
        self.precision = precision
        self.image_size = image_size
        self.num_classes_per_sample = num_classes_per_sample

        self.dataset_list = dataset.split("||")
        self.sample_rate = np.array(sample_rate or [1] * len(self.dataset_list))
        self.sample_rate /= self.sample_rate.sum()
        self.all_datasets = self.create_datasets()
        self.total_ds_len = sum(len(ds) for ds in self.all_datasets)

    def create_datasets(self):
        datasets = []
        for ds in self.dataset_list:
            dataset_cls = self.datasets_config.get(ds)
            if dataset_cls:
                if ds == "Semantic_Segm":
                    datasets.append(
                        dataset_cls(
                            self.dataset_dir,
                            self.tokenizer,
                            self.global_image_encoder,
                            self.epoch_samples,
                            self.precision,
                            self.image_size,
                            self.num_classes_per_sample,
                            self.semantic_segm_data,
                        )
                    )
                elif ds == "Refer_Segm":
                    datasets.append(
                        dataset_cls(
                            self.dataset_dir,
                            self.tokenizer,
                            self.global_image_encoder,
                            self.epoch_samples,
                            self.precision,
                            self.image_size,
                            self.num_classes_per_sample,
                            self.refer_segm_data,
                            use_qformer=self.use_qformer,
                        )
                    )
                elif ds == "Part_Segm":
                    datasets.append(
                        dataset_cls(
                            self.dataset_dir,
                            self.tokenizer,
                            self.global_image_encoder,
                            self.epoch_samples,
                            self.precision,
                            self.image_size,
                            self.num_classes_per_sample,
                            self.part_segm_data,
                            per_object_class_threshold=self.per_object_class_threshold,
                        )
                    )
                elif "CoSegm" in ds:
                    datasets.append(
                        dataset_cls(
                            self.dataset_dir,
                            self.tokenizer,
                            self.global_image_encoder,
                            self.epoch_samples,
                            self.precision,
                            self.image_size,
                            self.num_classes_per_sample,
                            self.cosegm_data,
                            multi_image_filepath_prefix=self.multi_image_filepath_prefix,
                            mode=self.mode,
                            use_qformer=self.use_qformer,
                        )
                    )
                elif "8Q" in ds:
                    datasets.append(
                        dataset_cls(
                            self.dataset_dir,
                            self.tokenizer,
                            self.global_image_encoder,
                            self.epoch_samples,
                            self.precision,
                            self.image_size,
                            self.num_classes_per_sample,
                            self.cosegm_data,
                            multi_image_filepath_prefix=self.multi_image_filepath_prefix,
                            mode=self.mode,
                            use_qformer=self.use_qformer,
                        )
                    )
                elif ds.startswith("Demo"):
                    datasets.append(
                        dataset_cls(
                            self.dataset_dir,
                            self.tokenizer,
                            self.global_image_encoder,
                            self.epoch_samples,
                            self.precision,
                            self.image_size,
                            self.num_classes_per_sample,
                            self.demo_data,
                            data_filepath_prefix=self.data_filepath_prefix,
                            mode=self.mode,
                        )
                    )
                else:
                    datasets.append(
                        dataset_cls(
                            self.dataset_dir,
                            self.tokenizer,
                            self.global_image_encoder,
                            self.epoch_samples,
                            self.precision,
                            self.image_size,
                            self.num_classes_per_sample,
                        )
                    )
        return datasets

    def __len__(self):
        return self.epoch_samples

    def _total_len(self):
        return self.total_ds_len

    def __getitem__(self, idx):
        dataset_idx = np.random.choice(len(self.dataset_list), p=self.sample_rate)
        selected_dataset = self.all_datasets[dataset_idx]
        data = selected_dataset[0]
        return (*data,)


class HybridCapDataset(HybridDatasetBase):
    def __init__(
        self,
        dataset_dir,
        tokenizer,
        global_image_encoder,
        epoch_samples=500 * 8 * 2 * 10,
        batch_size=2,
        precision="fp32",
        image_size=224,
        num_classes_per_sample=3,
        dataset="CocoCap||LLaVaInstruct",
        sample_rate=[1, 1],
    ):
        datasets_config = {
            # "CocoCap": CocoCapDataset,
            # "LLaVaInstruct": LLaVAInstructDataset,
            # "GrandCaptionDataset": GrandShortCaptionDataset,
            # Add other dataset mappings here
        }
        super().__init__(
            dataset_dir,
            tokenizer,
            global_image_encoder,
            dataset,
            datasets_config,
            epoch_samples,
            batch_size,
            precision,
            image_size,
            num_classes_per_sample,
            sample_rate,
        )


class HybridRegDataset(HybridDatasetBase):
    def __init__(
        self,
        dataset_dir,
        tokenizer,
        global_image_encoder,
        epoch_samples=500 * 8 * 2 * 10,
        batch_size=2,
        precision="fp32",
        image_size=224,
        num_classes_per_sample=3,
        dataset="RefCoco_Reg||RefCocoG_Reg||RefCocoP_Reg||VisGen_Reg||Flickr_Reg",
        sample_rate=[1, 1, 1, 1, 1],
    ):
        datasets_config = {
            # "RefCoco_Reg": RefCocoRegDataset,
            # "RefCocoG_Reg": RefCocoGRegDataset,
            # "RefCocoP_Reg": RefCocoPRegDataset,
            # "VisGen_Reg": VisualGenomeRegDataset,
            # "Flickr_Reg": Flickr30kRegDataset,
            # "GrandRefer_Reg": GrandReferRegDataset,
            # Add other dataset mappings here
        }
        super().__init__(
            dataset_dir,
            tokenizer,
            global_image_encoder,
            dataset,
            datasets_config,
            epoch_samples,
            batch_size,
            precision,
            image_size,
            num_classes_per_sample,
            sample_rate,
        )


class HybridSegDataset(HybridDatasetBase):
    def __init__(
        self,
        dataset_dir,
        tokenizer,
        global_image_encoder,
        epoch_samples=500 * 8 * 2 * 10,
        batch_size=2,
        precision="fp32",
        image_size=224,
        num_classes_per_sample=3,
        dataset="Semantic_Segm||Refer_Segm||PSG_GCG||RefCoco_GCG||GranDf_GCG||Flickr_GCG",
        sample_rate=[5, 4, 1, 1, 1, 1],
        semantic_segm_data="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        refer_segm_data="refcoco||refcocog||refcoco+||refclef",
        use_qformer=True,
    ):
        self.semantic_segm_data = semantic_segm_data
        self.refer_segm_data = refer_segm_data
        self.use_qformer = use_qformer
        datasets_config = {
            # "Semantic_Segm": SemanticSegmDataset,
            "Refer_Segm": ReferSegmDataset,
            # "PSG_GCG": OpenPsgGCGDataset,
            # "RefCoco_GCG": RefCOCOgGCGDataset,
            # "GranDf_GCG": GranDfDataset,
            # "Flickr_GCG": Flickr30kGCGDataset,
            # "GrandRefer_Segm": GrandReferSegmDataset,
            # Add other dataset mappings here
        }
        super().__init__(
            dataset_dir,
            tokenizer,
            global_image_encoder,
            dataset,
            datasets_config,
            epoch_samples,
            batch_size,
            precision,
            image_size,
            num_classes_per_sample,
            sample_rate,
        )

    # def __getitem__(self, idx):
    #     dataset_idx = np.random.choice(len(self.dataset_list), p=self.sample_rate)
    #     selected_dataset = self.all_datasets[dataset_idx]
    #     new_idxs = np.random.choice(
    #         len(selected_dataset), size=self.batch_size, replace=False
    #     )
    #     datas = [selected_dataset[new_idx] for new_idx in new_idxs]
    #     return datas

    def __getitem__(self, idx):
        dataset_idx = np.random.choice(len(self.dataset_list), p=self.sample_rate)
        selected_dataset = self.all_datasets[dataset_idx]
        idx = np.random.choice(len(selected_dataset))
        data = selected_dataset[idx]
        return (*data,)


class HybridPartDataset(HybridDatasetBase):
    def __init__(
        self,
        dataset_dir,
        tokenizer,
        global_image_encoder,
        epoch_samples=500 * 8 * 2 * 10,
        batch_size=2,
        precision="fp32",
        image_size=224,
        num_classes_per_sample=3,
        dataset="Part_Segm||Paired_Part_Segm||Object_CoSegm||Part_CoSegm||8Q_Single_Turn",
        sample_rate=[1, 1],
        part_segm_data="ade20k_part234||paco_lvis||part_image_net",
        cosegm_data="ade20k_part234||paco_lvis||pascal_part",
        per_object_class_threshold=10,
        multi_image_filepath_prefix=None,
        mode=None,
        use_qformer=True,
    ):
        self.part_segm_data = part_segm_data
        self.cosegm_data = cosegm_data
        self.use_qformer = use_qformer
        datasets_config = {
            # "Part_Segm": PartSegmDataset,
            # "Paired_Part_Segm": PairedPartSegmDataset,
            # "Object_CoSegm": MultiImageObjectComparisonDataset,
            # "Part_CoSegm": MultiImagePartComparisonDataset,
            # "8Q": MultiImage8QDataset,
            "8Q_Single_Turn": MultiImage8QSingleTurnDataset,
            # Add other dataset mappings here
        }
        self.per_object_class_threshold = per_object_class_threshold
        self.multi_image_filepath_prefix = multi_image_filepath_prefix
        self.mode = mode
        super().__init__(
            dataset_dir,
            tokenizer,
            global_image_encoder,
            dataset,
            datasets_config,
            epoch_samples,
            batch_size,
            precision,
            image_size,
            num_classes_per_sample,
            sample_rate,
        )

    # def __getitem__(self, idx):
    #     dataset_idx = np.random.choice(len(self.dataset_list), p=self.sample_rate)
    #     selected_dataset = self.all_datasets[dataset_idx]
    #     new_idxs = np.random.choice(
    #         len(selected_dataset), size=self.batch_size, replace=False
    #     )
    #     datas = [selected_dataset[new_idx] for new_idx in new_idxs]
    #     return datas

    def __getitem__(self, idx):
        dataset_idx = np.random.choice(len(self.dataset_list), p=self.sample_rate)
        selected_dataset = self.all_datasets[dataset_idx]
        idx = np.random.choice(len(selected_dataset))
        data = selected_dataset[idx]
        return (*data,)


class HybridDemoDataset(HybridDatasetBase):
    def __init__(
        self,
        dataset_dir,
        tokenizer,
        global_image_encoder,
        epoch_samples=500 * 8 * 2 * 10,
        batch_size=2,
        precision="fp32",
        image_size=224,
        num_classes_per_sample=3,
        dataset="DemoObjectID||DemoObjectIDSupercategory||DemoDifferentPartsImage||DemoDifferentPartsClass||DemoDifferentPartsImageClass||DemoPartLocalization||DemoWhy||DemoWhyNot",
        sample_rate=[1, 1, 1, 1, 1, 1, 1, 1],
        part_segm_data="ade20k_part234||paco_lvis||part_image_net",
        cosegm_data="ade20k_part234||paco_lvis||part_image_net",
        demo_data="ade20k_part234||paco_lvis||part_image_net||pascal_part",
        per_object_class_threshold=10,
        data_filepath_prefix=None,
        mode=None,
    ):
        self.part_segm_data = part_segm_data
        self.cosegm_data = cosegm_data
        self.demo_data = demo_data
        datasets_config = {
            # "DemoObjectID": ObjectIDDataset,
            # "DemoObjectIDSupercategory": ObjectIDSupercategoryDataset,
            # "DemoDifferentPartsImage": DifferentPartsImageDataset,
            # "DemoDifferentPartsClass": DifferentPartsClassDataset,
            # "DemoDifferentPartsImageClass": DifferentPartsImageClassDataset,
            # "DemoPartLocalization": PartLocalizationDataset,
            # "DemoWhy": WhyDataset,
            # "DemoWhyNot": WhyNotDataset,
        }
        self.per_object_class_threshold = per_object_class_threshold
        self.data_filepath_prefix = data_filepath_prefix
        self.mode = mode
        super().__init__(
            dataset_dir,
            tokenizer,
            global_image_encoder,
            dataset,
            datasets_config,
            epoch_samples,
            batch_size,
            precision,
            image_size,
            num_classes_per_sample,
            sample_rate,
        )

    # def __getitem__(self, idx):
    #     dataset_idx = np.random.choice(len(self.dataset_list), p=self.sample_rate)
    #     selected_dataset = self.all_datasets[dataset_idx]
    #     new_idx = np.random.choice(len(selected_dataset))
    #     data = selected_dataset[new_idx]
    #     return (*data,)

    def __getitem__(self, idx):
        dataset_idx = np.random.choice(len(self.dataset_list), p=self.sample_rate)
        selected_dataset = self.all_datasets[dataset_idx]
        new_idxs = np.random.choice(
            len(selected_dataset), size=self.batch_size, replace=False
        )
        datas = [selected_dataset[new_idx] for new_idx in new_idxs]
        return datas


# class HybridPartDataset(HybridDatasetBase):
#     def __init__(
#         self,
#         dataset_dir,
#         tokenizer,
#         global_image_encoder,
#         epoch_samples=500 * 8 * 2 * 10,
#         batch_size=2,
#         precision="fp32",
#         image_size=224,
#         num_classes_per_sample=3,
#         dataset="Part_Segm||Paired_Part_Segm",
#         sample_rate=[1, 1],
#         part_segm_data="ade20k_part234||paco_lvis||part_image_net",
#         cosegm_data="ade20k_part234||paco_lvis||part_image_net",
#         per_object_class_threshold=10,
#         paired_data_file_suffix=None,
#     ):
#         self.part_segm_data = part_segm_data
#         self.cosegm_data = cosegm_data
#         datasets_config = {
#             "Part_Segm": PartSegmDataset,
#             "Paired_Part_Segm": PairedPartSegmDataset,
#             # Add other dataset mappings here
#         }
#         self.per_object_class_threshold = per_object_class_threshold
#         self.paired_data_file_suffix = paired_data_file_suffix
#         super().__init__(
#             dataset_dir,
#             tokenizer,
#             global_image_encoder,
#             dataset,
#             datasets_config,
#             epoch_samples,
#             batch_size,
#             precision,
#             image_size,
#             num_classes_per_sample,
#             sample_rate,
#         )

#         def alternative_len(self):
#             return self._total_len()

#         if self.epoch_samples < self._total_len():
#             self.__len__ = alternative_len

#     def __getitem__(self, idx):
#         # dataset_idx = np.random.choice(len(self.dataset_list), p=self.sample_rate)
#         # selected_dataset = self.all_datasets[dataset_idx]
#         # idx = idx % len(selected_dataset)
#         # data = selected_dataset[idx]
#         idx = idx % self._total_len()
#         cum_len = 0
#         for ds in self.all_datasets:
#             if cum_len + len(ds) > idx:
#                 actual_index = idx - cum_len
#                 data = ds[actual_index]
#                 return (*data,)
#             cum_len += len(ds)
#         raise IndexError(
#             f"Index {idx} out of bounds, total dataset length is {self._total_len()}"
#         )


def custom_collate_fn(
    batch,
    tokenizer=None,
    use_mm_start_end=True,
    inference=False,
    local_rank=-1,
    img_emb_len=32,
):
    # Initializing lists and counters
    image_path_list, global_enc_image_list, grounding_enc_image_list = [], [], []
    bboxes_list, conversation_list, masks_list = [], [], []
    label_list, resize_list, questions_list = [], [], []
    selected_labels_list, offset_list, inferences = [], [0], []
    cnt = 0

    # Iterating through the batch
    for (
        image_path,
        global_enc_image,
        grounding_enc_image,
        bboxes,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
    ) in batch:
        image_path_list.append(image_path)
        global_enc_image_list.append(global_enc_image)
        grounding_enc_image_list.append(grounding_enc_image)
        bboxes_list.append(bboxes)
        conversation_list.extend(conversations)
        masks_list.append([] if masks is None else masks.float())
        label_list.append(label)
        resize_list.append(resize)
        questions_list.append(questions)
        selected_labels_list.append(sampled_classes)
        offset_list.append(cnt := cnt + len(conversations))
        inferences.append(inference)

    # Handling the conversation list
    if use_mm_start_end:
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        conversation_list = [
            conv.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            for conv in conversation_list
        ]

    # Tokenizing and padding input ids
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [
            tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            for prompt in conversation_list
        ],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    # Preparing targets and handling conversation types
    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()
    # conv_type == "llava_v1"
    sep = conv.sep + conv.roles[1] + ": "
    sep2 = conv.sep2

    for conversation, target in zip(conversation_list, targets):
        _process_conversation(conversation, target, tokenizer, sep, sep2)

    # Adjusting for inferences
    if not inferences[0]:
        truncate_len = tokenizer.model_max_length - (img_emb_len - 1)
        if input_ids.shape[1] > truncate_len:
            input_ids, targets, attention_masks = map(
                lambda x: x[:, :truncate_len], [input_ids, targets, attention_masks]
            )

    return {
        "image_paths": image_path_list,
        "global_enc_images": torch.stack(global_enc_image_list, dim=0),
        "grounding_enc_images": (
            None
            if grounding_enc_image_list[0] is None
            else torch.stack(grounding_enc_image_list, dim=0)
        ),
        "bboxes": None if bboxes_list[0] is None else bboxes_list,
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": None if masks_list[0] is None else masks_list,
        "label_list": None if label_list[0] is None else label_list,
        "resize_list": None if resize_list[0] is None else resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": selected_labels_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }


def custom_collate_fn_grouped_val(
    batch,
    tokenizer=None,
    use_mm_start_end=True,
    inference=False,
    local_rank=-1,
    img_emb_len=32,
):
    # Initializing lists and counters
    image_path_list, global_enc_image_list, grounding_enc_image_list = [], [], []
    correspondence_feats_list = []
    bboxes_list, conversation_list, masks_list = [], [], []
    label_list, resize_list, questions_list = [], [], []
    selected_labels_list, offset_list, img_offset_list, inferences = [], [0], [0], []
    cnt = 0
    img_cnt = 0

    # if len(batch) > 1:
    #     print(
    #         "Warning: batch size should be 1 for this collate function. Instead got "
    #         f"inner_batch_size = {len(batch[0])} and batch_size = {len(batch)}. Only "
    #         "first element in batch processes, discarding others."
    #     )

    # Iterating through the batch
    for (
        image_path,
        global_enc_image,
        grounding_enc_image,
        bboxes,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
    ) in batch:
        correspondence_feats = None
        if isinstance(global_enc_image, (tuple, list)):
            global_enc_image, correspondence_feats = global_enc_image
        if image_path is None:
            image_path = []
        elif not isinstance(image_path, (tuple, list)):
            image_path = [image_path]
            global_enc_image = global_enc_image.unsqueeze(0)
            if correspondence_feats is not None:
                correspondence_feats = correspondence_feats.unsqueeze(0)
            grounding_enc_image = grounding_enc_image.unsqueeze(0)
            bboxes = [bboxes] if bboxes is not None else None
        image_path_list.append(image_path)
        if global_enc_image is not None:
            global_enc_image_list.append(global_enc_image)
        if correspondence_feats is not None:
            correspondence_feats_list.append(correspondence_feats)
        if grounding_enc_image is not None:
            grounding_enc_image_list.append(grounding_enc_image)
        bboxes_list.append(bboxes)
        conversation_list.extend(conversations)
        if masks is None:
            masks_list.append(None)
        elif isinstance(masks, (tuple, list)):
            masks_list.append(
                tuple((mask.float() if mask is not None else None) for mask in masks)
            )
        else:
            masks_list.append(masks.float())
        label_list.append(label)
        resize_list.append(resize)
        questions_list.append(questions)
        selected_labels_list.append(sampled_classes)
        offset_list.append(cnt := cnt + len(conversations))
        if global_enc_image is not None:
            img_offset_list.append(img_cnt := img_cnt + global_enc_image.shape[0])
        else:
            img_offset_list.append(img_cnt)
        inferences.append(inference)

    # Handling the conversation list
    if use_mm_start_end:
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        conversation_list = [
            conv.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            for conv in conversation_list
        ]

    # Tokenizing and padding input ids
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [
            tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            for prompt in conversation_list
        ],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    # Preparing targets and handling conversation types
    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()
    # conv_type == "llava_v1"
    sep = conv.sep + conv.roles[1] + ": "
    sep2 = conv.sep2

    for conversation, target in zip(conversation_list, targets):
        _process_conversation(conversation, target, tokenizer, sep, sep2)

    # Adjusting for inferences
    if not inferences[0]:
        truncate_len = tokenizer.model_max_length - (img_emb_len - 1) * len(
            image_path_list[0]
        )
        # print(f"{input_ids.shape[1] = }, max = {input_ids.shape[1] + (img_emb_len - 1) * 2}")
        if input_ids.shape[1] > truncate_len:
            print(
                f"Warning: input_ids too long ({input_ids.shape[1]}), truncating to {truncate_len}. Conversations:\n"
                + conversation_list
            )
            input_ids, targets, attention_masks = map(
                lambda x: x[:, :truncate_len], [input_ids, targets, attention_masks]
            )

    # try:
    if len(global_enc_image_list) > 0:
        global_enc_images_ret = torch.cat(global_enc_image_list, dim=0)
    else:
        global_enc_images_ret = None

    if len(correspondence_feats_list) > 0 and correspondence_feats_list[0] is not None:
        global_enc_images_ret = (
            global_enc_images_ret,
            torch.cat(correspondence_feats_list, dim=0),
        )

    return {
        "image_paths": image_path_list,
        "global_enc_images": global_enc_images_ret,
        "grounding_enc_images": (
            None
            if len(grounding_enc_image_list) == 0 or grounding_enc_image_list[0] is None
            else torch.cat(grounding_enc_image_list, dim=0)
        ),
        "bboxes": None if bboxes_list[0] is None else bboxes_list,
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": None if masks_list[0] is None else masks_list,
        "label_list": None if label_list[0] is None else label_list,
        "resize_list": None if resize_list[0] is None else resize_list,
        "offset": torch.LongTensor(offset_list),
        "image_offset": torch.LongTensor(img_offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": selected_labels_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }


def custom_collate_fn_grouped(
    batch,
    tokenizer=None,
    use_mm_start_end=True,
    inference=False,
    local_rank=-1,
    img_emb_len=32,
):
    # Initializing lists and counters
    image_path_list, global_enc_image_list, grounding_enc_image_list = [], [], []
    correspondence_feats_list = []
    bboxes_list, conversation_list, masks_list = [], [], []
    label_list, resize_list, questions_list = [], [], []
    selected_labels_list, offset_list, img_offset_list, inferences = [], [0], [0], []
    cnt = 0
    img_cnt = 0

    # if len(batch) > 1:
    #     print(
    #         "Warning: batch size should be 1 for this collate function. Instead got "
    #         f"inner_batch_size = {len(batch[0])} and batch_size = {len(batch)}. Only "
    #         "first element in batch processes, discarding others."
    #     )

    # Iterating through the batch
    for (
        image_path,
        global_enc_image,
        grounding_enc_image,
        bboxes,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
    ) in batch[0]:
        correspondence_feats = None
        if isinstance(global_enc_image, (tuple, list)):
            global_enc_image, correspondence_feats = global_enc_image
        if image_path is None:
            image_path = []
        elif not isinstance(image_path, (tuple, list)):
            image_path = [image_path]
            global_enc_image = global_enc_image.unsqueeze(0)
            if correspondence_feats is not None:
                correspondence_feats = correspondence_feats.unsqueeze(0)
            grounding_enc_image = grounding_enc_image.unsqueeze(0)
            bboxes = [bboxes] if bboxes is not None else None
        image_path_list.append(image_path)
        if global_enc_image is not None:
            global_enc_image_list.append(global_enc_image)
        if correspondence_feats is not None:
            correspondence_feats_list.append(correspondence_feats)
        if grounding_enc_image is not None:
            grounding_enc_image_list.append(grounding_enc_image)
        bboxes_list.append(bboxes)
        conversation_list.extend(conversations)
        if masks is None:
            masks_list.append(None)
        elif isinstance(masks, (tuple, list)):
            masks_list.append(
                tuple((mask.float() if mask is not None else None) for mask in masks)
            )
        else:
            masks_list.append(masks.float())
        label_list.append(label)
        resize_list.append(resize)
        questions_list.append(questions)
        selected_labels_list.append(sampled_classes)
        offset_list.append(cnt := cnt + len(conversations))
        if global_enc_image is not None:
            img_offset_list.append(img_cnt := img_cnt + global_enc_image.shape[0])
        else:
            img_offset_list.append(img_cnt)
        inferences.append(inference)

    # Handling the conversation list
    if use_mm_start_end:
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        conversation_list = [
            conv.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            for conv in conversation_list
        ]

    # Tokenizing and padding input ids
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [
            tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            for prompt in conversation_list
        ],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    # Preparing targets and handling conversation types
    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()
    # conv_type == "llava_v1"
    sep = conv.sep + conv.roles[1] + ": "
    sep2 = conv.sep2

    for conversation, target in zip(conversation_list, targets):
        _process_conversation(conversation, target, tokenizer, sep, sep2)

    # Adjusting for inferences
    if not inferences[0]:
        truncate_len = tokenizer.model_max_length - (img_emb_len - 1) * len(
            image_path_list[0]
        )
        # print(f"{input_ids.shape[1] = }, max = {input_ids.shape[1] + (img_emb_len - 1) * 2}")
        if input_ids.shape[1] > truncate_len:
            print(
                f"Warning: input_ids too long ({input_ids.shape[1]}), truncating to {truncate_len}. Conversations:\n"
                + conversation_list
            )
            input_ids, targets, attention_masks = map(
                lambda x: x[:, :truncate_len], [input_ids, targets, attention_masks]
            )

    # try:
    if len(global_enc_image_list) > 0:
        global_enc_images_ret = torch.cat(global_enc_image_list, dim=0)
    else:
        global_enc_images_ret = None

    if len(correspondence_feats_list) > 0 and correspondence_feats_list[0] is not None:
        global_enc_images_ret = (
            global_enc_images_ret,
            torch.cat(correspondence_feats_list, dim=0),
        )

    new_masks_list, new_label_list, new_resize_list = [], [], []

    return {
        "image_paths": image_path_list,
        "global_enc_images": global_enc_images_ret,
        "grounding_enc_images": (
            None
            if len(grounding_enc_image_list) == 0 or grounding_enc_image_list[0] is None
            else torch.cat(grounding_enc_image_list, dim=0)
        ),
        "bboxes": None if bboxes_list[0] is None else bboxes_list,
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": None if masks_list[0] is None else masks_list,
        "label_list": None if label_list[0] is None else label_list,
        "resize_list": None if resize_list[0] is None else resize_list,
        "offset": torch.LongTensor(offset_list),
        "image_offset": torch.LongTensor(img_offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": selected_labels_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }


def custom_collate_fn_8q(
    batch,
    tokenizer=None,
    use_mm_start_end=True,
    inference=False,
    local_rank=-1,
    img_emb_len=32,
    qformer_tokenizer=None,
):
    # Initializing lists and counters
    image_path_list, global_enc_image_list, grounding_enc_image_list = [], [], []
    bboxes_list, conversation_list, masks_list = [], [], []
    label_list, resize_list, questions_list = [], [], []
    selected_labels_list, offset_list, img_offset_list, inferences = [], [0], [0], []
    cnt = 0
    img_cnt = 0

    # if len(batch) > 1:
    #     print(
    #         "Warning: batch size should be 1 for this collate function. Instead got "
    #         f"inner_batch_size = {len(batch[0])} and batch_size = {len(batch)}. Only "
    #         "first element in batch processes, discarding others."
    #     )

    # Iterating through the batch
    for batch_idx, (
        image_path,
        global_enc_image,
        grounding_enc_image,
        bboxes,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
    ) in enumerate(batch):
        if not image_path:
            image_path = []
            assert not masks and not label and not resize and not bboxes
        elif not isinstance(image_path, (tuple, list)):
            image_path = [image_path]
            global_enc_image = global_enc_image.unsqueeze(0)
            grounding_enc_image = grounding_enc_image.unsqueeze(0)
            bboxes = [bboxes] if bboxes is not None else None
            label = [label] if label is not None else None
            masks = [masks] if masks is not None else None
            resize = [resize] if resize is not None else None
        image_path_list.append(image_path)
        if global_enc_image is not None:
            global_enc_image_list.append(global_enc_image)
        if grounding_enc_image is not None:
            grounding_enc_image_list.append(grounding_enc_image)
        bboxes_list.append(bboxes)
        conversation_list.extend(conversations)
        if not masks:
            masks_list.append(None)
        else:
            assert isinstance(masks, (tuple, list))
            masks_list.append(
                [(mask.float() if mask is not None else None) for mask in masks]
            )
        label_list.append(label)
        resize_list.append(resize)
        questions_list.extend(questions)
        selected_labels_list.append(sampled_classes)
        offset_list.append(cnt := cnt + len(conversations))
        if global_enc_image is not None:
            img_offset_list.append(img_cnt := img_cnt + global_enc_image.shape[0])
        else:
            img_offset_list.append(img_cnt)
        inferences.append(inference)

    # Handling the conversation list
    if use_mm_start_end:
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        conversation_list = [
            conv.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            for conv in conversation_list
        ]

    # Tokenizing and padding input ids
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [
            tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            for prompt in conversation_list
        ],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    # qformer_inputs = qformer_tokenizer(
    #     questions_list, padding=True, truncation=True, return_tensors="pt"
    # )
    qformer_input_ids = None  # qformer_inputs.input_ids
    qformer_attention_masks = None  # qformer_inputs.attention_mask

    # Preparing targets and handling conversation types
    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()
    # conv_type == "llava_v1"
    sep = conv.sep + conv.roles[1] + ": "
    sep2 = conv.sep2

    for conversation, target in zip(conversation_list, targets):
        _process_conversation(conversation, target, tokenizer, sep, sep2)

    # Adjusting for inferences
    if not inferences[0]:
        truncate_len = tokenizer.model_max_length - (img_emb_len - 1) * len(
            image_path_list[0]
        )
        # print(f"{input_ids.shape[1] = }, max = {input_ids.shape[1] + (img_emb_len - 1) * 2}")
        if input_ids.shape[1] > truncate_len:
            print(
                f"Warning: input_ids too long ({input_ids.shape[1]}), truncating to {truncate_len}. Conversations:\n"
                + conversation_list
            )
            input_ids, targets, attention_masks = map(
                lambda x: x[:, :truncate_len], [input_ids, targets, attention_masks]
            )

    # try:
    if len(global_enc_image_list) > 0:
        global_enc_images_ret = torch.cat(global_enc_image_list, dim=0)
    else:
        global_enc_images_ret = None

    new_masks_list = list(zip_longest(*masks_list))
    new_label_list = list(zip_longest(*label_list))
    new_resize_list = list(zip_longest(*resize_list))

    return {
        "image_paths": image_path_list,
        "global_enc_images": global_enc_images_ret,
        "grounding_enc_images": (
            None
            if len(grounding_enc_image_list) == 0 or grounding_enc_image_list[0] is None
            else torch.cat(grounding_enc_image_list, dim=0)
        ),
        "bboxes": None if bboxes_list[0] is None else bboxes_list,
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": new_masks_list,  # None if masks_list[0] is None else masks_list,
        "label_list": new_label_list,  # None if label_list[0] is None else label_list,
        "resize_list": new_resize_list,  # None if resize_list[0] is None else resize_list,
        "offset": torch.LongTensor(offset_list),
        "image_offset": torch.LongTensor(img_offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": selected_labels_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        "qformer_input_ids": qformer_input_ids,
        "qformer_attention_masks": qformer_attention_masks,
    }


def _process_conversation(conversation, target, tokenizer, sep, sep2):
    total_len = target.ne(tokenizer.pad_token_id).sum().item()
    rounds = conversation.split(sep2)
    cur_len = 1
    target[:cur_len] = IGNORE_INDEX

    for rou in rounds:
        if not rou:
            break

        parts = rou.split(sep)
        # assert len(parts) == 2, (len(parts), rou)
        if len(parts) == 1:
            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids) - 1
            target[cur_len : cur_len + round_len] = IGNORE_INDEX
            cur_len += round_len
            continue
        parts[0] += sep

        if DEFAULT_IMAGE_TOKEN in conversation:
            round_len = len(tokenizer_image_token(rou, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
        else:
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

        target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
        cur_len += round_len

    target[cur_len:] = IGNORE_INDEX
    if cur_len < tokenizer.model_max_length:
        assert cur_len == total_len, f"{cur_len = }, {total_len = }"

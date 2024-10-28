import os
import json
import re
from dataset.utils.coco import COCO, maskUtils


def load_json_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


# from .question_templates.captioning import (
#     caption_templates,
#     region_templates,
#     region_group_templates,
# )
# from .question_templates.grounded_conversation import gcg_instructions
# from .question_templates.segmentation import (
#     seg_templates,
#     seg_region_templates,
#     seg_id_templates,
#     seg_id_region_templates,
#     part_templates,
#     part_region_templates,
#     part_id_templates,
#     part_id_region_templates,
# )
# from .question_templates.multi_image import (
#     # multi_image_id_object_templates,
#     multi_image_common_object_templates,
#     multi_image_common_parts_templates,
#     multi_image_unique_parts_templates,
# )

# from .question_templates.helper_methods import (
#     get_all_questions,
#     get_all_questions_with_regions,
#     get_prepend_questions,
# )

# # Captioning

# # CAPTION_QUESTIONS = get_all_questions(caption_templates)
# CAPTION_QUESTIONS = caption_templates

# # # Region-level captioning

# # REGION_QUESTIONS = get_all_questions(region_templates)
# REGION_QUESTIONS = region_templates

# # # Region group captioning

# # REGION_GROUP_QUESTIONS = get_all_questions(region_group_templates)
# REGION_GROUP_QUESTIONS = region_group_templates

# # # Grounded conversation generation

# # gcg_templates = [
# #     f"{question} {instruction}"
# #     for question in caption_templates
# #     for instruction in gcg_instructions
# # ]

# # GCG_QUESTIONS = get_all_questions(gcg_templates)
# gcg_templates = [
#     f"{question} {instruction}"
#     for question in caption_templates
#     for instruction in gcg_instructions
# ]
# GCG_QUESTIONS = gcg_templates

# # # Segmentation

# # SEG_QUESTIONS = get_all_questions(seg_templates)
# # SEG_REGION_QUESTIONS = get_all_questions(seg_region_templates)
# # SEG_ID_QUESTIONS = get_all_questions(seg_id_templates)
# # SEG_ID_REGION_QUESTIONS = get_all_questions(seg_id_region_templates)

# SEG_REGION_QUESTIONS = seg_region_templates

# # SEG_ANSWER_LIST = [
# #     "The {class_name} is [SEG].",
# #     "The segmentation result for the {class_name} is [SEG].",
# # ]

# # # Part questions

# # PART_QUESTIONS = get_all_questions(part_templates)
# # PART_REGION_QUESTIONS = get_all_questions(part_region_templates)
# # PART_ID_QUESTIONS = get_all_questions(part_id_templates)
# # PART_ID_REGION_QUESTIONS = get_all_questions(part_id_region_templates)

# # PART_ANSWER_LIST = [
# #     ("The {class_name} has", "a"),
# #     ("The parts of the {class_name} are", "the"),
# #     ("The parts of the {class_name} include", "the"),
# #     ("The parts of the {class_name} consist of", "the"),
# #     ("The {class_name}'s parts are", "the"),
# #     ("The {class_name}'s parts include", "the"),
# #     ("The {class_name}'s parts consist of", "the"),
# # ]

# # # Multi-image localization
# # MULTI_IMAGE_ID_OBJECT_QUESTIONS = get_all_questions(
# #     multi_image_id_object_templates, num_images=2
# # )
# # MULTI_IMAGE_ID_OBJECT_REGION_QUESTIONS = get_all_questions_with_regions(
# #     multi_image_id_object_templates, num_images=2
# # )
# ID_OBJECT_ANSWER_LIST = {
#     "same": [
#         "Both images show {a_class_name}.",
#         "The images depict {a_class_name}.",
#         "Each image contains {a_class_name}.",
#         "Both objects are {a_class_name}.",
#         "The pictures show {a_class_name}.",
#         "The images include {a_class_name}.",
#         "Both photos feature {a_class_name}.",
#         "Each picture depicts {a_class_name}.",
#         "The images have {a_class_name}.",
#         "The photos show {a_class_name}.",
#     ],
#     "different": [
#         "The images show {a_class_names}.",
#         "The pictures depict {a_class_names}.",
#         "The objects in the images are {a_class_names}.",
#         "The photos feature {a_class_names}.",
#         "The pictures include {a_class_names}.",
#         "The photos depict {a_class_names}.",
#         "The images contain {a_class_names}.",
#     ],
# }

# # MULTI_IMAGE_OBJECT_QUESTIONS = get_all_questions(
# #     multi_image_common_object_templates, num_images=2
# # )
# # MULTI_IMAGE_OBJECT_REGION_QUESTIONS = get_all_questions_with_regions(
# #     multi_image_common_object_templates, num_images=2
# # )
# # COMMON_OBJECT_ANSWER_LIST = {
# #     "singular": [
# #         ("The common object is {the_class_name}.", "the")
# #     ],
# #     "plural": [
# #         ("The common objects are {the_class_names}.", "the")
# #     ],
# # }


# # fmt: off
# MULTI_IMAGE_OBJECT_QUESTIONS = multi_image_common_object_templates
# COMMON_OBJECT_ANSWER_LIST = {
#     "singular": [
#         "The common object is {the_class_names}.",
#         "There is one common object: {the_class_names}.",
#         "The shared object is {the_class_names}.",
#         "Identified common object: {the_class_names}.",
#         "The recurring object is {the_class_names}.",
#         "The common object found is {the_class_names}.",
#         "The object present in the images is {the_class_names}.",
#         "The object appearing in these pictures is {the_class_names}.",
#         "The detected common object is {the_class_names}.",
#         "The same object across the images is {the_class_names}.",

#         "The images show {a_class_names}.",
#         "There is {a_class_names} in these images.",
#         "These images depict {a_class_names}.",
#         "Each image contains {a_class_names}.",
#         "A common object in the images is {a_class_names}.",
#         "The images include {a_class_names}.",
#         "You can see {a_class_names} in the images.",
#         "There appears to be {a_class_names} in each photo.",
#         "The detected object is {a_class_names}.",
#         "Each picture shows {a_class_names}.",
#     ],
#     "plural": [
#         "The common objects are {the_class_names}.",
#         "There are multiple common objects: {the_class_names}.",
#         "The shared objects are {the_class_names}.",
#         "Identified common objects: {the_class_names}.",
#         "The recurring objects are {the_class_names}.",
#         "The common objects found are {the_class_names}.",
#         "The objects present in the images are {the_class_names}.",
#         "The objects appearing in these pictures are {the_class_names}.",
#         "The detected common objects are {the_class_names}.",
#         "The same objects across the images are {the_class_names}.",

#         "The images show {a_class_names}.",
#         "There are {a_class_names} in these images.",
#         "These images depict {a_class_names}.",
#         "Each image contains {a_class_names}.",
#         "Common objects in the images are {a_class_names}.",
#         "The images include {a_class_names}.",
#         "You can see {a_class_names} in the images.",
#         "There appear to be {a_class_names} in each photo.",
#         "The detected objects are {a_class_names}.",
#         "Each picture shows {a_class_names}.",
#     ],
# }
# # fmt: on

# # MULTI_IMAGE_COMMON_PART_QUESTIONS = get_all_questions(
# #     multi_image_common_parts_templates, num_images=2
# # )
# # MULTI_IMAGE_COMMON_PART_REGION_QUESTIONS = get_all_questions_with_regions(
# #     multi_image_common_parts_templates, num_images=2
# # )
# # MULTI_IMAGE_COMMON_PART_ANSWER_LIST = {
# #     "singular": [("The common part between the objects is", "the")],
# #     "plural": [("The common parts between the objects are", "the")],
# # }

# MULTI_IMAGE_COMMON_PART_QUESTIONS = multi_image_common_parts_templates
# MULTI_IMAGE_COMMON_PART_ANSWER_LIST = {
#     "singular": [
#         "The common part between the objects is {the_parts}.",
#         "The shared part is {the_parts}.",
#         "Both objects have {the_parts}.",
#         "The recurring part is {the_parts}.",
#         "Identified common part: {the_parts}.",
#         "The same part in both objects is {the_parts}.",
#         "There is one common part: {the_parts}.",
#         "The objects share {the_parts}.",
#         "Common part found: {the_parts}.",
#         "The detected part is {the_parts}.",
#         "The common part between the objects is {a_parts}.",
#         "The shared part is {a_parts}.",
#         "Both objects share {a_parts}.",
#         "The recurring part is {a_parts}.",
#         "Identified common part: {a_parts}.",
#         "The same part in both objects is {a_parts}.",
#         "There is one common part: {a_parts}.",
#         "The objects share {a_parts}.",
#         "Common part found: {a_parts}.",
#         "The detected part is {a_parts}.",
#     ],
#     "plural": [
#         "The common parts between the objects are {the_parts}.",
#         "The shared parts are {the_parts}.",
#         "Both objects have {the_parts}.",
#         "The recurring parts are {the_parts}.",
#         "Identified common parts: {the_parts}.",
#         "The same parts in both objects are {the_parts}.",
#         "There are multiple common parts: {the_parts}.",
#         "The objects share {the_parts}.",
#         "Common parts found: {the_parts}.",
#         "The detected parts are {the_parts}.",
#         "The common parts between the objects are {a_parts}.",
#         "The shared parts are {a_parts}.",
#         "Both objects share {a_parts}.",
#         "The recurring parts are {a_parts}.",
#         "Identified common parts: {a_parts}.",
#         "The same parts in both objects are {a_parts}.",
#         "There are multiple common parts: {a_parts}.",
#         "The objects share {a_parts}.",
#         "Common parts found: {a_parts}.",
#         "The detected parts are {a_parts}.",
#     ],
# }

# # MULTI_IMAGE_UNIQUE_PART_QUESTIONS = get_all_questions(
# #     multi_image_unique_parts_templates, num_images=2
# # )
# # MULTI_IMAGE_UNIQUE_PART_REGION_QUESTIONS = get_all_questions_with_regions(
# #     multi_image_unique_parts_templates, num_images=2
# # )
# # MULTI_IMAGE_UNIQUE_PART_ANSWER_LIST = [
# #     ("The unique parts of the objects are", "the"),
# #     # ("The unique parts of the object in {image_name} are", "the", "while"),
# # ]

# MULTI_IMAGE_UNIQUE_PART_QUESTIONS = multi_image_unique_parts_templates
# MULTI_IMAGE_UNIQUE_PART_ANSWER_LIST = {
#     "singular": [
#         "The unique part between the objects is {the_parts}.",
#         "The distinct part is {the_parts}.",
#         "The unique part identified is {the_parts}.",
#         "There is one unique part: {the_parts}.",
#         "The distinct part found is {the_parts}.",
#         "The unique part present is {the_parts}.",
#         "The detected unique part is {the_parts}.",
#         "The singular unique part is {the_parts}.",
#     ],
#     "plural": [
#         "The unique parts between the objects are {the_parts}.",
#         "The distinct parts are {the_parts}.",
#         "The unique parts identified are {the_parts}.",
#         "The objects have {the_parts} as unique features.",
#         "There are multiple unique parts: {the_parts}.",
#         "The distinct parts found are {the_parts}.",
#         "The unique parts present are {the_parts}.",
#         "The detected unique parts are {the_parts}.",
#         "The multiple unique parts are {the_parts}.",
#         "The unique parts between the objects are {a_parts}.",
#         "The distinct parts are {a_parts}.",
#         "The unique parts identified are {a_parts}.",
#         "There are multiple unique parts: {a_parts}.",
#         "The distinct parts found are {a_parts}.",
#         "The unique parts present are {a_parts}.",
#         "The detected unique parts are {a_parts}.",
#         "The multiple unique parts are {a_parts}.",
#     ],
# }


SEG_QUESTIONS = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",
    "Can you segment the {class_name} in this image",
    "Please segment {class_name} in this image",
    "What is {class_name} in this image? Please respond with segmentation mask",
    "What is {class_name} in this image? Please output segmentation mask",
    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",
    "Could you provide a segmentation mask for the {class_name} in this image",
    "Please identify and segment the {class_name} in this image",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask",
    "Can you highlight the {class_name} in this image with a segmentation mask",
]

SEG_ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]


def _add_article(word, article):
    assert article.strip().lower() in [
        "a",
        "an",
        "the",
    ], f"unsupported article: {article}"
    article = "an" if "a" in article and word[0] in "aeiou" else article
    return f"{article.strip()} {word.strip()}"


def _format(template, **kwargs):
    def replace(matchobj):
        key = matchobj.group(1)
        value = str(kwargs[key.lower()])
        if key[0].isupper():
            if value.startswith("<p>"):
                rest = value[len("<p>") :]
                for i, char in enumerate(rest):
                    if char.isalpha():
                        rest = rest[:i] + char.capitalize() + rest[i + 1 :]
                        break
                return "<p>" + rest
            return str(kwargs[key.lower()]).capitalize()
        return str(kwargs[key])

    pattern = re.compile(r"{(\w+)}")
    result = pattern.sub(replace, template)
    return result


def _write_list(items):
    if len(items) == 0:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def init_ade20k_part234(dataset_dir, mode, area_threshold=None):
    assert mode in ["train", "val"], f"Invalid mode for ADE20K Part 234: {mode}"
    ade20k_part234_api = COCO(
        os.path.join(
            dataset_dir, "ADE20KPart234", f"ade20k_instance_{mode}_mixed_parts.json"
        ),
        area_threshold,
    )
    all_obj_classes = ade20k_part234_api.loadCats(ade20k_part234_api.getCatIds())
    class_map_ade20k_part234 = {"object": {}, "part": {}}
    for cat in all_obj_classes:
        name = cat["name"]
        class_map_ade20k_part234["object"][cat["id"]] = name

    all_part_classes = load_json_file(
        "dataset/utils/ade20k_part234_classes_mixed_parts.json"
    )
    for idx, cat in enumerate(all_part_classes):
        obj, part = cat.split("'s ")
        class_map_ade20k_part234["part"][idx] = (obj, part)

    img_ids = ade20k_part234_api.getImgIds()
    return class_map_ade20k_part234, img_ids, ade20k_part234_api


def init_paco_lvis(dataset_dir, mode, area_threshold=None):
    assert mode in ["train", "val", "test"], f"Invalid mode for PACO LVIS: {mode}"
    paco_lvis_api = COCO(
        os.path.join(
            dataset_dir, "paco_lvis", "annotations", f"paco_lvis_v1_{mode}.json"
        ),
        area_threshold,
    )
    all_classes = paco_lvis_api.loadCats(paco_lvis_api.getCatIds())
    class_map_paco_lvis = {"object": {}, "part": {}}

    for cat in all_classes:
        cat_split = cat["name"].strip().split(":")
        if len(cat_split) == 1:
            name = cat_split[0].split("_(")[0]
            name = " ".join(name.split("_"))
            class_map_paco_lvis["object"][cat["id"]] = name
        else:
            assert len(cat_split) == 2
            obj, part = cat_split
            obj = obj.split("_(")[0]
            part = part.split("_(")[0]
            obj = " ".join(obj.split("_"))
            part = " ".join(part.split("_"))
            name = (obj, part)
            class_map_paco_lvis["part"][cat["id"]] = name
        # class_map_paco_lvis[cat["id"]] = name

    img_ids = paco_lvis_api.getImgIds()
    return class_map_paco_lvis, img_ids, paco_lvis_api


def init_pascal_part(dataset_dir, mode, area_threshold=None):
    assert mode in ["train", "val"], f"Invalid mode for PASCAL-Part: {mode}"
    pascal_part_api = COCO(
        os.path.join(dataset_dir, "PASCAL-Part", f"{mode}.json"),
        area_threshold,
    )
    all_classes = pascal_part_api.loadCats(pascal_part_api.getCatIds())
    class_map_pascal_part = {"object": {}, "part": {}}

    for cat in all_classes:
        cat_split = cat["name"].strip().split(":")
        if len(cat_split) == 1:
            name = cat_split[0].split("_(")[0]
            name = " ".join(name.split("_"))
            class_map_pascal_part["object"][cat["id"]] = name
        else:
            assert len(cat_split) == 2
            obj, part = cat_split
            obj = obj.split("_(")[0]
            part = part.split("_(")[0]
            obj = " ".join(obj.split("_"))
            part = " ".join(part.split("_"))
            name = (obj, part)
            class_map_pascal_part["part"][cat["id"]] = name

    img_ids = pascal_part_api.getImgIds()
    return class_map_pascal_part, img_ids, pascal_part_api


def init_part_image_net(dataset_dir, mode, area_threshold=None):
    assert mode in ["train", "val", "test"], f"Invalid mode for PartImageNet: {mode}"
    pin_api_part = COCO(
        os.path.join(
            dataset_dir, "PartImageNet", "annotations", mode, f"{mode}_mixed_parts.json"
        ),
        area_threshold,
    )
    pin_api_obj = COCO(
        os.path.join(
            dataset_dir,
            "PartImageNet",
            "annotations",
            f"{mode}_whole",
            f"{mode}_mixed_parts.json",
        ),
        area_threshold,
    )
    all_part_classes = pin_api_part.loadCats(pin_api_part.getCatIds())
    all_obj_classes = pin_api_obj.loadCats(pin_api_obj.getCatIds())
    class_map_pin = {"object": {}, "part": {}, "object_to_supercategory": {}}
    for cat in all_part_classes:
        cat_main, cat_part = cat["supercategory"], cat["name"]
        name = (cat_main, cat_part)
        class_map_pin["part"][cat["id"]] = name
    for cat in all_obj_classes:
        name = " ".join(cat["name"].lower().split("_"))
        class_map_pin["object"][cat["id"]] = name
        class_map_pin["object_to_supercategory"][name] = cat["supercategory"]
    img_ids = pin_api_part.getImgIds()
    return class_map_pin, img_ids, (pin_api_obj, pin_api_part)


def segm_to_mask(segm, h, w):
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm["counts"]) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    m = maskUtils.decode(rle)
    return m

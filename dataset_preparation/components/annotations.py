from typing import Dict, Tuple, List

import pandas as pd


def load_annotations(annotations_csv_filepath: str) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """
    From a PHOENIX14T dataset annotations CSV file,
    return:
        - examples IDs in a list
        - dictionary of the form `{ExampleID: {'gloss': ExampleGloss, 'text': ExampleText}}`
    """
    df = pd.read_csv(annotations_csv_filepath, delimiter="|")
    df = df.rename(columns={"name": "id", "orth": "gloss", "translation": "text"})
    return list(df.id), df.set_index("id")[["gloss", "text"]].to_dict(orient="index")


def write_files_file(examples_ids: List[str], subset: str, dataset_folder: str):
    """
    Write dataset's `subset`.files file.
    """
    with open(f"{dataset_folder}/{subset}.files", "w", encoding="utf-8") as f:
        f.writelines(f"{subset}/{example_id}" + "\n" for example_id in examples_ids)


def write_text_file(
        annotations: Dict[str, Dict[str, str]],
        examples_ids: List[str],
        subset: str,
        dataset_folder: str
):
    """
    Write dataset's `subset`.text file performing necessary preprocessing on text.
    Should write lowercase sentences with a wit a dot at the end. Example: 'my name is jean .'
    Will be written only for `example_ids` examples, in the same order as of the given list.
    """
    with open(f"{dataset_folder}/{subset}.text", "w", encoding="utf-8") as f:
        f.writelines(annotations[example_id]["text"].lower() + " .\n" for example_id in examples_ids)


def write_gloss_file(
        annotations: Dict[str, Dict[str, str]],
        examples_ids: List[str],
        subset: str,
        dataset_folder: str
):
    """
    Write dataset's `subset`.gloss file.
    Example of a line: 'I NAME JEAN'
    Will be written only for `example_ids` examples, in the same order as of the given list.
    """
    with open(f"{dataset_folder}/{subset}.gloss", "w", encoding="utf-8") as f:
        f.writelines(f"{annotations[example_id]["gloss"]}\n" for example_id in examples_ids)

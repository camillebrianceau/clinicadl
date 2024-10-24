import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from clinicadl.utils.tsvtools_utils import extract_baseline
from tests.testing_tools import compare_folders

"""
Check the absence of data leakage
    1) Baseline datasets contain only one scan per subject
    2) No intersection between train and test sets
    3) Absence of MCI train subjects in test sets of subcategories of MCI
"""


@pytest.fixture(
    params=[
        "test_getlabels",
        "test_split",
        "test_analysis",
        "test_get_progression",
        "test_prepare_experiment",
        "test_get_metadata",
    ]
)
def test_name(request):
    return request.param


def test_tsvtools(cmdopt, tmp_path, test_name):
    if test_name == "test_getlabels":
        run_test_getlabels(cmdopt, tmp_path)
    elif test_name == "test_split":
        run_test_split(cmdopt, tmp_path)
    elif test_name == "test_analysis":
        run_test_analysis(cmdopt, tmp_path)
    elif test_name == "test_prepare_experiment":
        run_test_prepare_experiment(cmdopt, tmp_path)
    elif test_name == "test_get_progression":
        run_test_get_progression(cmdopt, tmp_path)
    elif test_name == "test_get_metadata":
        run_test_get_metadata(cmdopt, tmp_path)


def check_is_subject_unique(labels_path_baseline: Path):
    print("Check subject uniqueness", labels_path_baseline)

    flag_is_unique = True
    check_df = pd.read_csv(labels_path_baseline, sep="\t")
    check_df.set_index(["participant_id", "session_id"], inplace=True)
    if labels_path_baseline.name[-12:] != "baseline.tsv":
        check_df = extract_baseline(check_df, set_index=False)
    for _, subject_df in check_df.groupby(level=0):
        if len(subject_df) > 1:
            flag_is_unique = False
    assert flag_is_unique


def check_is_independant(
    train_path_baseline: Path, test_path_baseline: Path, subject_flag=True
):
    print("Check independence")

    flag_is_independant = True
    train_df = pd.read_csv(train_path_baseline, sep="\t")
    train_df.set_index(["participant_id", "session_id"], inplace=True)
    test_df = pd.read_csv(test_path_baseline, sep="\t")
    test_df.set_index(["participant_id", "session_id"], inplace=True)

    for subject, session in train_df.index:
        if (subject, session) in test_df.index:
            flag_is_independant = False

    assert flag_is_independant


def run_test_suite(data_tsv, n_splits):
    check_train = True
    if n_splits == 0:
        train_baseline_tsv = data_tsv / "train_baseline.tsv"
        test_baseline_tsv = data_tsv / "test_baseline.tsv"
        if not train_baseline_tsv.exists():
            check_train = False

        check_is_subject_unique(test_baseline_tsv)
        if check_train:
            check_is_subject_unique(train_baseline_tsv)
            check_is_independant(train_baseline_tsv, test_baseline_tsv)

    else:
        for split_number in range(n_splits):
            for folder, _, files in os.walk(data_tsv / "split"):
                folder = Path(folder)

                for file in files:
                    if file[-3:] == "tsv":
                        check_is_subject_unique(folder / file)
                train_baseline_tsv = folder / "train_baseline.tsv"
                test_baseline_tsv = folder / "test_baseline.tsv"
                if train_baseline_tsv.exists():
                    if test_baseline_tsv.exists():
                        check_is_independant(train_baseline_tsv, test_baseline_tsv)


def run_test_getlabels(cmdopt, tmp_path):
    """Checks that getlabels is working and that it is coherent with
    previous version in reference_path."""

    base_dir = Path(cmdopt["input"])
    input_dir = base_dir / "tsvtools" / "in"
    ref_dir = base_dir / "tsvtools" / "ref"
    tmp_out_dir = tmp_path / "tsvtools" / "out"
    tmp_out_dir.mkdir(parents=True)

    import shutil

    bids_output = tmp_out_dir / "bids"
    bids_directory = input_dir / "bids"
    restrict_tsv = input_dir / "restrict.tsv"
    output_tsv = tmp_out_dir
    if tmp_out_dir.exists():
        shutil.rmtree(tmp_out_dir)
        tmp_out_dir.mkdir(parents=True)
    shutil.copytree(bids_directory, bids_output)
    merged_tsv = input_dir / "merge-tsv.tsv"
    missing_mods_directory = input_dir / "missing_mods"

    flag_getlabels = not os.system(
        f"clinicadl -vvv tsvtools get-labels {str(bids_output)} {str(output_tsv)} "
        f"-d AD -d MCI -d CN -d Dementia "
        f"--merged_tsv {str(merged_tsv)} --missing_mods {str(missing_mods_directory)} "
        f"--restriction_tsv {str(restrict_tsv)}"
    )
    assert flag_getlabels

    out_df = pd.read_csv(tmp_out_dir / "labels.tsv", sep="\t")
    ref_df = pd.read_csv(ref_dir / "labels.tsv", sep="\t")
    assert out_df.equals(ref_df)


def run_test_split(cmdopt, tmp_path):
    """Checks that:
    -  split and kfold are working
    -  the loading functions can find the output
    -  no data leakage is introduced in split and kfold.
    """

    base_dir = Path(cmdopt["input"])
    input_dir = base_dir / "tsvtools" / "in"
    ref_dir = base_dir / "tsvtools" / "ref"
    tmp_out_dir = tmp_path / "tsvtools" / "out"
    tmp_out_dir.mkdir(parents=True)

    n_test = 10
    n_splits = 2
    train_tsv = tmp_out_dir / "split/train.tsv"
    labels_tsv = tmp_out_dir / "labels.tsv"
    shutil.copyfile(input_dir / "labels.tsv", labels_tsv)

    flag_split = not os.system(
        f"clinicadl -vvv tsvtools split {str(labels_tsv)} --subset_name test --n_test {n_test}"
    )
    flag_getmetadata = not os.system(
        f"clinicadl -vvv tsvtools get-metadata {str(train_tsv)} {str(labels_tsv)} -voi age -voi sex -voi diagnosis"
    )
    flag_kfold = not os.system(
        f"clinicadl -vvv tsvtools kfold {str(train_tsv)} --n_splits {n_splits} --subset_name validation"
    )
    assert flag_split
    assert flag_getmetadata
    assert flag_kfold

    assert compare_folders(tmp_out_dir / "split", ref_dir / "split", tmp_out_dir)

    run_test_suite(tmp_out_dir, n_splits)


def run_test_analysis(cmdopt, tmp_path):
    """Checks that analysis can be performed."""

    base_dir = Path(cmdopt["input"])
    input_dir = base_dir / "tsvtools" / "in"
    ref_dir = base_dir / "tsvtools" / "ref"
    tmp_out_dir = tmp_path / "tsvtools" / "out"
    tmp_out_dir.mkdir(parents=True)

    merged_tsv = input_dir / "merge-tsv.tsv"
    labels_tsv = input_dir / "labels.tsv"
    output_tsv = tmp_out_dir / "analysis.tsv"
    ref_analysis_tsv = ref_dir / "analysis.tsv"

    flag_analysis = not os.system(
        f"clinicadl tsvtools analysis {str(merged_tsv)} {str(labels_tsv)} {str(output_tsv)} "
        f"--diagnoses CN --diagnoses MCI --diagnoses Dementia"
    )

    assert flag_analysis
    ref_df = pd.read_csv(ref_analysis_tsv, sep="\t")
    out_df = pd.read_csv(output_tsv, sep="\t")
    assert out_df.equals(ref_df)


def run_test_get_progression(cmdopt, tmp_path):
    """Checks that get-progression can be performed"""

    base_dir = Path(cmdopt["input"])
    input_dir = base_dir / "tsvtools" / "in"
    ref_dir = base_dir / "tsvtools" / "ref"
    tmp_out_dir = tmp_path / "tsvtools" / "out"
    tmp_out_dir.mkdir(parents=True)

    input_progression_tsv = input_dir / "labels.tsv"
    progression_tsv = tmp_out_dir / "progression.tsv"
    ref_progression_tsv = ref_dir / "progression.tsv"
    shutil.copyfile(input_progression_tsv, progression_tsv)

    flag_get_progression = not os.system(
        f"clinicadl tsvtools get-progression {str(progression_tsv)}  "
    )
    assert flag_get_progression

    ref_df = pd.read_csv(ref_progression_tsv, sep="\t")
    out_df = pd.read_csv(progression_tsv, sep="\t")
    assert out_df.equals(ref_df)


def run_test_prepare_experiment(cmdopt, tmp_path):
    """Checks that:
    -  split and kfold are working
    -  the loading functions can find the output
    -  no data leakage is introduced in split and kfold.
    """

    base_dir = Path(cmdopt["input"])
    input_dir = base_dir / "tsvtools" / "in"
    ref_dir = base_dir / "tsvtools" / "ref"
    tmp_out_dir = tmp_path / "tsvtools" / "out"
    tmp_out_dir.mkdir(parents=True)

    labels_tsv = tmp_out_dir / "labels.tsv"
    shutil.copyfile(input_dir / "labels.tsv", labels_tsv)

    validation_type = "kfold"
    n_valid = 2
    n_test = 10
    flag_prepare_experiment = not os.system(
        f"clinicadl -vvv tsvtools prepare-experiment {str(labels_tsv)} --n_test {n_test} --validation_type {validation_type} --n_validation {n_valid}"
    )

    assert flag_prepare_experiment

    assert compare_folders(tmp_out_dir / "split", ref_dir / "split", tmp_out_dir)

    run_test_suite(tmp_out_dir, n_valid)


def run_test_get_metadata(cmdopt, tmp_path):
    """Checks that get-metadata can be performed"""
    base_dir = Path(cmdopt["input"])
    input_dir = base_dir / "tsvtools" / "in"
    ref_dir = base_dir / "tsvtools" / "ref"
    tmp_out_dir = tmp_path / "tsvtools" / "out"
    tmp_out_dir.mkdir(parents=True)

    input_metadata_tsv = input_dir / "restrict.tsv"
    metadata_tsv = tmp_out_dir / "metadata.tsv"
    input_labels_tsv = input_dir / "labels.tsv"
    labels_tsv = tmp_out_dir / "labels.tsv"
    ref_metadata_tsv = ref_dir / "metadata.tsv"

    shutil.copyfile(input_metadata_tsv, metadata_tsv)
    shutil.copyfile(input_labels_tsv, labels_tsv)

    flag_get_metadata = not os.system(
        f"clinicadl tsvtools get-metadata {str(metadata_tsv)} {str(labels_tsv)} -voi diagnosis -voi sex -voi age"
    )
    assert flag_get_metadata

    ref_df = pd.read_csv(ref_metadata_tsv, sep="\t")
    out_df = pd.read_csv(metadata_tsv, sep="\t")
    assert out_df.equals(ref_df)

import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

MIMICCXR_SECTIONED_CSV = '<path to mimic-cxr folder>/mimic_cxr_sectioned.csv'
MIMICCXR_MASTER_CSV = '<path to mimic-cxr folder>/mimic-cxr-2.0.0-chexpert.csv'
MIMICCXR_SPLIT_CSV = '<path to mimic-cxr folder>/mimic-cxr-2.0.0-split.csv'
MIMICCXR_METADATA_CSV = '<path to mimic-cxr folder>/mimic-cxr-2.0.0-metadata.csv'
MIMICCXR_IMG_PATH = '<path to mimic-cxr images>'
MIMICCXR_IMG_PATH_INSERT = ''
ONLY_FRONTAL = False
MIMICCXR_TRAIN_CSV = '<path to output folder>/train.csv'
MIMICCXR_VAL_CSV = '<path to output folder>/val.csv'
MIMICCXR_TEST_CSV = '<path to output folder>/test.csv'


def create_im_path(subject_id, study_id, dicom_id):
    str_subjectid = str(subject_id)
    return os.path.join(MIMICCXR_IMG_PATH_INSERT, 'p' + str_subjectid[0:2], 'p' + str_subjectid, 's' + str(study_id),
                        dicom_id + '.jpg')

def create_report_mimic(df_sectioned, study_id):
    full_report = ''
    rval = 0
    if ('s' + str(study_id)) not in df_sectioned["study"].to_numpy():
        print('s' + str(study_id) + ' no report')
        full_report = ''  # np.nan
        rval = 1  # no study id in csv
    else:
        df_tmp = df_sectioned[df_sectioned["study"] == ('s' + str(study_id))]
        # for section in ['impression', 'findings', 'last_paragraph']:
        for section in ['findings', 'last_paragraph', 'impression']:
            if section == 'impression' and full_report != '':
                continue
            txt = df_tmp[section].item()
            if txt == txt:
                full_report = full_report + txt
        if len(full_report) < 3:
            rval = 2
    return full_report, rval


def preprocess_mimiccxr_data():
    # use the original splits
    df_split = pd.read_csv(MIMICCXR_SPLIT_CSV)
    df_split = df_split.drop(columns=['study_id', 'subject_id'])

    # filter the desired pathologies
    df_pathologies = pd.read_csv(MIMICCXR_MASTER_CSV)
    df_pathologies = df_pathologies.drop(columns=['subject_id'])
    pathologies = list(df_pathologies.columns)
    pathologies = pathologies[1:]

    df_sectioned = pd.read_csv(MIMICCXR_SECTIONED_CSV)

    # get views
    df_metadata = pd.read_csv(MIMICCXR_METADATA_CSV)
    df_metadata = df_metadata.drop(columns=['StudyDate', 'PerformedProcedureStepDescription', 'StudyTime',
                                            'ProcedureCodeSequence_CodeMeaning', 'ViewCodeSequence_CodeMeaning',
                                            'PatientOrientationCodeSequence_CodeMeaning'])

    df_metadata_frontal = df_metadata[(df_metadata["ViewPosition"] == "PA") | (df_metadata["ViewPosition"] == "AP")]
    df_metadata_lateral = df_metadata[
        (df_metadata["ViewPosition"] == "LATERAL") | (df_metadata["ViewPosition"] == "LL")]

    if ONLY_FRONTAL:
        df_new = pd.merge(df_pathologies, df_metadata_frontal, how='inner', on='study_id')
    else:
        df_new = pd.merge(df_pathologies, df_metadata, how='inner', on='study_id')
    df_new = pd.merge(df_new, df_split, how='inner', on='dicom_id')
    df_new = df_new.reset_index(drop=True)

    # add path column to df_master
    im_path_all = len(df_new) * ['']
    reports_all = len(df_new) * ['']
    ind_remove = []
    no_reports_cnt = 0
    short_reports_cnt = 0
    for index, row in tqdm(df_new.iterrows(), total=len(df_new)):
        im_path = create_im_path(row.subject_id, row.study_id, row.dicom_id)
        im_path_all[index] = im_path
        report, rval = create_report_mimic(df_sectioned, row.study_id)
        if rval == 1:
            no_reports_cnt += 1
        elif rval == 2:
            short_reports_cnt += 1
        # report = create_report(row.subject_id_x, row.study_id)
        if len(report) < 3 or not os.path.exists(os.path.join(MIMICCXR_IMG_PATH, im_path)):
            ind_remove.append(index)
        reports_all[index] = report

    df_new['Path'] = im_path_all
    df_new['Report'] = reports_all
    df_new['N_pathologies'] = len(df_new) * [len(pathologies)]
    print(f"Original # of samples: {len(df_new)}")
    df_new = df_new.drop(ind_remove)
    print(f"Samples removed: {len(ind_remove)}")
    print(f"# not existing reports: {no_reports_cnt}")
    print(f"# too short reports: {short_reports_cnt}")
    print(f"New # of samples: {len(df_new)}")

    valid_df = df_new.loc[df_new['split'] == 'validate']
    test_df = df_new.loc[df_new['split'] == 'test']
    train_df = df_new.loc[df_new['split'] == 'train']

    print(f"Number of train samples: {len(train_df)}")
    print(f"Number of valid samples: {len(valid_df)}")
    print(f"Number of test samples: {len(test_df)}")

    os.makedirs('/'.join(MIMICCXR_TRAIN_CSV.split('/')[:-1]), exist_ok=True)
    train_df.to_csv(MIMICCXR_TRAIN_CSV)
    valid_df.to_csv(MIMICCXR_VAL_CSV)
    test_df.to_csv(MIMICCXR_TEST_CSV)


def available_datasets():
    """Returns the names of available datasets"""
    return list(_DATASETS.keys())


_DATASETS = {
    # "chexpert": preprocess_chexpert_data,
    "mimiccxr": preprocess_mimiccxr_data
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="dataset type",
        default="mimiccxr",
    )
    args = parser.parse_args()

    if args.dataset.lower() in _DATASETS.keys():
        _DATASETS[args.dataset.lower()]()
    else:
        RuntimeError(
            f"Model {args.dataset} not found; available datasets = {available_datasets()}"
        )

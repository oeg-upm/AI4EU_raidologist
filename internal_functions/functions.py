import datetime
import os
import pickle
import random
import re
import string
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import spacy
import torch as t
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from imageio import imread
from scispacy.abbreviation import AbbreviationDetector
from torch.autograd import Variable
from torchtext import data
import sys
from urllib.request import urlopen
from urllib.parse import quote
import json
import pydicom
import shutil
from pydicom.pixel_data_handlers.util import apply_color_lut
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import internal_functions


def _save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def _load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


##INDICATOR 1: Image Processing
def _get_kaze_vector(image_file, roi=None):
    """Generates a feature vector for the image using the KAZE keypoint detection method.
     INPUT: Image file, [Coordinates of ROI in a list [x,y,w,h]]
     OUTPUT: Descriptor vector"""
    # Check the extension of the image file
    extension = image_file.rstrip().split('.')[-1]
    if extension == 'dcm':
        dicom_image = pydicom.dcmread(image_file)
        arr = dicom_image.pixel_array
        rgb_image = apply_color_lut(arr, dicom_image, palette='PET')
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        scale_percent = 60  # percent of original size
        width = int(gray.shape[1] * scale_percent / 100)
        height = int(gray.shape[0] * scale_percent / 100)
        dim = (width, height)
        gray = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)

    else:
        image = imread(image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if roi:
        gray = gray[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    alg = cv2.KAZE_create()
    kps = alg.detect(gray)
    kps = sorted(kps, key=lambda x: -x.response)[:128]
    kps, dsc = alg.compute(gray, kps)
    if dsc is None:
        return kps, np.zeros((512,))
    dsc = dsc.flatten()
    if dsc.size < 512:
        # if we have less the 32 descriptors then just adding zeros at the
        # end of our feature vector
        dsc = np.concatenate([dsc, np.zeros(512 - dsc.size)])
    elif dsc.size > 512:
        n = 64
        while dsc.size > 512:
            n = int(n / 2)
            kps = sorted(kps, key=lambda x: -x.response)[:n]
            kps, dsc = alg.compute(gray, kps)
            dsc = dsc.flatten()
    return dsc


def _get_neural_embedding(image_file, roi=None):
    """Generates a feature vector for a given image using a pretrained convolutional neural network (ResNet18)
    INPUT: Image file, [Coordinates of ROI in a list [x,y,w,h]]
    OUTPUT: Feature vector of the image"""
    cnn_model = models.resnet18(pretrained=True)
    # Feature layer
    layer = cnn_model._modules.get('avgpool')
    cnn_model.eval()
    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    extension = image_file.rstrip().split('.')[-1]
    if extension == 'dcm':
        if extension == 'dcm':
            dicom_image = pydicom.dcmread(image_file)
            arr = dicom_image.pixel_array
            img = apply_color_lut(arr, dicom_image, palette='PET')
        if roi:
            img = img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        scale_percent = 60  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        img_array = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        img = Image.fromarray(img_array)
    else:
        img = Image.open(image_file)
        if roi:
            img = img.crop(roi)
    pytorch_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    feature_vector = t.zeros(512)

    def copy_vector(m, i, o):
        feature_vector.copy_(o.data.squeeze())

    h = layer.register_forward_hook(copy_vector)
    cnn_model(pytorch_img)
    return np.array(feature_vector.data)


def _query_SNOMED(term, i=0, query_type='concept'):
    baseUrl = 'https://browser.ihtsdotools.org/snowstorm/snomed-ct'
    edition = 'MAIN'
    version = '2020-03-09'
    equivalence_cliner_dict = {'disease': 'problem', 'procedure': 'test', 'substance': 'treatment'}
    ret = 'N/A'
    if query_type == 'concept':
        url = baseUrl + '/browser/' + edition + '/' + version + '/descriptions?&limit=50&term=' + quote(term) \
              + '&conceptActive=true&lang=english&skipTo=0&returnLimit=100'
        try:
            response = urlopen(url).read()
            snomed_data = json.loads(response.decode('utf-8'))
            if snomed_data['totalElements'] >= 1:
                ret = snomed_data['items'][i]['concept']['fsn']['term']
        except:
            pass
    elif query_type == 'entity_type':
        url = baseUrl + '/browser/' + edition + '/' + version + '/descriptions?&limit=50&term=' + quote(term) \
              + '&conceptActive=true&lang=english&skipTo=0&returnLimit=100'
        try:
            response = urlopen(url).read()
            snomed_data = json.loads(response.decode('utf-8'))
            if snomed_data['totalElements'] >= 1:
                s = snomed_data['items'][i]['concept']['fsn']['term']
                ret = s[s.find("(") + 1:s.find(")")]
                if ret in equivalence_cliner_dict.keys():
                    ret = equivalence_cliner_dict[ret]
        except:
            pass
    return ret


#######FUNCTIONS FOR MODEL TRAINING
def _prepare_data(folder):
    """Parses the input files (from a given format) into a CSV file
    INPUT: Folder containing the input files, [Format. Default is XML]"""
    w = open('externals/tmp/dataset.csv', 'w+')
    w.write('text,label,\n')
    for filename in os.listdir(folder):
        name = filename.split('.')[0]
        case = _load_obj(os.path.join(folder, name))
        report = case.get_solution().get_section_report()
        labeled_report = section_string_to_dict(report.rstrip().split('\n'))
        for k, v in labeled_report.items():
            if len(v) > 0:
                w.write("\"" + v + '\",' + k + ',\n')
    w.close()


def train_section_model(case_folder, params=None):
    """Trains a section formatting model. If no specific parameters are specified, the best identified values are used.
    OUTPUT: Trained model, text vocabulary and label vocabulary"""
    _prepare_data(case_folder)
    if params is None:
        params = {'embedding_dim': 100, 'num_hidden_nodes': 32, 'num_output_nodes': 5, 'bidirection': True,
                  'num_layers': 2, 'dropout': 0.2}
    t.backends.cudnn.deterministic = True
    TEXT = data.Field(tokenize='spacy', batch_first=True, include_lengths=True)
    LABEL = data.LabelField(dtype=t.long, batch_first=True)
    fields = [('text', TEXT), ('label', LABEL)]
    training_data = data.TabularDataset(path='externals/tmp/dataset.csv', format='csv', fields=fields,
                                        skip_header=True)
    train_data, valid_data = training_data.split(split_ratio=0.15, random_state=random.seed(2020))
    TEXT.build_vocab(training_data, min_freq=1, vectors="glove.6B.100d")
    LABEL.build_vocab(training_data)
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    BATCH_SIZE = 32
    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train_data, valid_data),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        device=device)
    size_of_vocab = len(TEXT.vocab)
    params['size_of_vocab'] = size_of_vocab
    model = internal_functions.classifier(size_of_vocab, params['embedding_dim'], params['num_hidden_nodes'],
                                          params['num_output_nodes'],
                                          params['num_layers'], bidirectional=params['bidirection'],
                                          dropout=params['dropout'])

    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    model, optimizer, criterion = internal_functions.optimizer_and_loss(model, device)

    N_EPOCHS = 10
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        model, train_loss, train_acc = internal_functions.train(model, train_iterator, optimizer, criterion)

        valid_loss, valid_acc = internal_functions.evaluate(model, valid_iterator, criterion)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            _save_obj({'params': params, 'model': model.state_dict(), 'vocab_dict': TEXT.vocab.stoi,
                       'label_dict': LABEL.vocab.stoi, 'acc': valid_acc, 'timestamp': datetime.datetime.utcnow()},
                      'externals/tmp/section_model')

        # print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        # print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
    os.remove('externals/tmp/dataset.csv')
    if not os.path.exists('externals/'+device.type+'_section_model.pkl'):
        shutil.move('externals/tmp/section_model.pkl', 'externals/'+device.type+'_section_model.pkl')
    return model, TEXT.vocab.stoi, LABEL.vocab.stoi


######PUBLIC FUNCTIONS

###Retrieval functions

def image_feature_extraction(image_file, roi=None):
    """Generates a feature vector for the given image.
    INPUT: Image file, [Select_roi (True if the user wants to select a ROI ad-hoc, false otherwise), ROI Coordinates of the image in the format [x,y,w,h]]
    OUTPUT: Feature vector of the image"""
    wb_vector = _get_kaze_vector(image_file, roi)[1]
    bb_vector = _get_neural_embedding(image_file, roi)
    # We combine both vectors by averaging them:
    feature_vector = np.mean([wb_vector, bb_vector], axis=0)
    return feature_vector


# INDICATOR 2: Document embedding
def get_document_embedding(text):
    """Generates the document embedding of a given report
    INPUT: Textual data
    OUTPUT: Text embedding"""
    spacy.prefer_gpu()
    nlp = spacy.load('en_core_sci_md')
    doc = nlp(text)
    embedding = doc.vector
    del nlp
    return embedding


# INDICATOR 3: NER
def get_entities(text, named_entities=None, prefix=''):
    """Detects the existing named entities within the text. Entities are stored in lowercase format to enable string comparison
    INPUT: Textual data, [Recognized named entities]
    OUTPUT: Returns a dictionary with the identified named entities and their type"""
    # Cat text into a file
    ner_file = prefix + 'externals/tmp/report_ner.txt'
    with open(ner_file, 'w+') as txt_file:
        txt_file.write(text)
    txt_file.close()
    # Call CliNER
    os.system(
        'python ' + prefix + 'externals/CliNER/cliner predict --txt ' + prefix +
        'externals/tmp/report_ner.txt --out ' + prefix + 'externals/tmp --model ' + prefix + 'externals/CliNER/models/silver.crf --format i2b2')
    os.remove(ner_file)
    # Parse output
    found_entities = {}
    p1 = re.compile('c="([^"]*)"')
    p2 = re.compile('t="([^"]*)"')
    output_ner = prefix + 'externals/tmp/report_ner.con'
    with open(output_ner, 'r') as ner_file:
        for line in ner_file:
            if line != '\n':
                c = p1.findall(line)[0]
                t = p2.findall(line)[0]
                found_entities[c.lower()] = t
    os.remove(output_ner)
    if named_entities:
        cat_ner = {}
        for ne in named_entities:
            cat = _query_SNOMED(ne, query_type='entity_type')
            if cat != 'N/A':
                cat_ner[ne.lower()] = cat
        found_entities = {**found_entities, **cat_ner}
    return found_entities


# INDICATOR 4: Abbreviatures
def get_abbr_ratio(text, known_abbreviatures=None):
    """Returns the percentage of dissambiguated abbreviations of the text.
    INPUT: Textual data, [Abbreviatures already identified]
    OUTPUT: Percentage of identified abbreviatures"""
    spacy.prefer_gpu()
    nlp = spacy.load('en_core_sci_md')
    abbreviation_pipe = AbbreviationDetector(nlp)
    nlp.add_pipe(abbreviation_pipe)
    doc = nlp(text)
    tokens = list(set([t.text for t in doc if t.text not in string.punctuation]))
    abbrs = [d.text for d in doc._.abbreviations]
    if len(abbrs) == 0:
        return 1
    if known_abbreviatures:
        for ka in known_abbreviatures:
            if ka in abbrs: abbrs.remove(ka)
    return float(len(list(set(abbrs) & set(tokens)))) / float(len(abbrs))


###Solution-related functions
def section_text(input_report):
    """Formats a given text into the four main sections.
    INPUT: Report to be formatted
    OUTPUT: Returns a formatted version of the input report
    """
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    if not os.path.exists('externals/'+device.type+'_section_model.pkl'):
        return None
    model_data = _load_obj('externals/'+device.type+'_section_model')
    params = model_data['params']
    model = internal_functions.classifier(params['size_of_vocab'], params['embedding_dim'],
                                          params['num_hidden_nodes'],
                                          params['num_output_nodes'],
                                          params['num_layers'], bidirectional=params['bidirection'],
                                          dropout=params['dropout'])
    model.load_state_dict(model_data['model'])
    vocab = model_data['vocab_dict']
    labels = model_data['label_dict']
    labels = {y: x for x, y in labels.items()}
    model.eval()
    # First, we split our report into sentences.
    sentences = input_report.rstrip().split('.')
    # Then, we will predict over each sentence:
    predictions = {}
    for s in sentences:
        if len(s) > 1:
            predictions[s] = labels[internal_functions.predict_sentence(model, vocab, s)]
    ret = ""
    for v in list(set(predictions.values())):
        ret += '###' + v + '\n'
        for k in predictions.keys():
            if predictions[k] == v:
                ret += k + '.'
        ret += '\n'
    return ret


def disambiguate_abbreviation(input_report):
    """Provides disambiguation suggestions for the unidentified abbreviations in the text.
    The suggestions are extracted from SNOMED-CT terminology
    INPUT: Input Report
    OUTPUT: A dictionary containing the best match disambiguation per identified abbreviation"""
    spacy.prefer_gpu()
    nlp = spacy.load('en_core_sci_md')
    doc = nlp(input_report)
    potential_abbreviations = list(
        set([t.text for t in doc if (t.text not in string.punctuation) and (t.text.isupper() or
                                                                             t.text not in doc.vocab)]))
    dissambiguated_abbreviations = {}
    exceptions = ['FINDINGS', 'COMPARISON', 'INDICATION', 'IMPRESSION', 'XXXX']
    potential_abbreviations = list(set(potential_abbreviations) - set(exceptions))
    for pa in potential_abbreviations:
        dissambiguated_abbreviations[pa] = _query_SNOMED(pa)
    abbreviations={k:v for k,v in dissambiguated_abbreviations.items() if v!='N/A'}
    return abbreviations


def return_related_entities(case_entities):
    """Given the entities detected in the report, suggests other related entities grouped by type
    (disease, procedure, test)
    INPUT: Set of entities detected in related cases
    OUTPUT: Entities detected in related cases grouped by type and without duplicates"""
    suggested_entities = {}
    suggested_entities['problem'] = list(set([k for k, v in case_entities.items() if v == 'problem']))
    suggested_entities['test'] = list(set([k for k, v in case_entities.items() if v == 'test']))
    suggested_entities['treatment'] = list(set([k for k, v in case_entities.items() if v == 'treatment']))
    return suggested_entities


def get_case_related_entities(st, related_cases):
    """Returns the all the entities featured in the related cases without duplicates.
    INPUT: Storage unit, related cases.
    OUTPUT: All the entities and their types contained in the related cases without duplicates"""
    existing_ners = {}
    for c in related_cases:
        case_ents = st.get_case_entities(c)
        ents = {}
        for k, v in case_ents.items():
            ents.update(v)
        if not set(existing_ners) & set(ents):
            for k, e in ents.items():
                existing_ners[k] = e
    return existing_ners


def section_string_to_dict(sectioned_report):
    """Parses a string containing the sectioned report into a dictionary containing each section and its content
    INPUT: String containing a sectioned report
    OUTPUT: Dictionary containing each section with its content"""
    sectioned_report_to_dict = {}
    labels = ['###FINDINGS', '###COMPARISON', '###INDICATION', '###IMPRESSION', 'XXXX',
              'FINDINGS', 'COMPARISON', 'INDICATION', 'IMPRESSION']
    cur_key = ''
    cur_section = ''
    for line in sectioned_report:
        line = line.rstrip().lstrip()
        if line in labels:
            if cur_section != '':
                if cur_key.startswith("#"):
                    sectioned_report_to_dict[cur_key[3:]] = cur_section
                else:
                    sectioned_report_to_dict[cur_key] = cur_section
                cur_key = line
                cur_section = ''
            else:
                cur_key = line
        else:
            if line != '\r': cur_section += line
    if cur_key.startswith("#"):
        sectioned_report_to_dict[cur_key[3:]] = cur_section
    else:
        sectioned_report_to_dict[cur_key] = cur_section
    return sectioned_report_to_dict


# Scoring model
def generate_scoring_model_data(case_index):
    """Generates the training and evaluation data for the scoring model"""
    df = pd.read_pickle(case_index)
    labeled_rows = df[(df.Validation_Status.isin(['Validated', 'Rejected']))]
    embeddings = labeled_rows['Doc_Embedding'].to_list()
    labels = labeled_rows['Validation_Status'].to_list()
    labels[labels == 'Validated'] = 1
    labels[labels == 'Rejected'] = 0
    x_train, x_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.25)
    return [[x_train, y_train], [x_test, y_test]]


def train_scoring_model(training_data, test_data):
    """Trains a scoring model """
    model = RandomForestClassifier()
    model.fit(training_data[0], training_data[1])
    acc = model.score(test_data[0], test_data[1])
    _save_obj({'model': model, 'acc': acc, 'timestamp': datetime.datetime.utcnow()}, 'externals/scoring_model')
    return model,acc


def score_case(input_text):
    """Scores an input report.
    INPUT: Report to evaluate in textual format
    OUTPUT: Prediction, Confidence value"""
    model=_load_obj('externals/scoring_model')
    predicting_model=model['model']
    embedding = get_document_embedding(input_text)
    predictions=predicting_model.predict_proba(np.reshape(embedding,(-1,embedding.shape[0])))
    if predictions[0].argmax()==0:
        return {'result':'Rejected','confidence':predictions[0][0]}
    else:
        return {'result':'Validated','confidence':predictions[0][1]}


import itertools
import sys
import os
import xml.etree.ElementTree as ET
import datetime
import pickle
import pandas as pd
import numpy as np
from scipy import spatial
import operator
from ftplib import FTP
import pysftp
import shutil

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import internal_functions



# De momento vamos a centrarnos en local y ya daremos el salto


def _save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def _load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def _ftp_upload(ftp_data, path, object, remotefile, extension='.pkl'):
    ftp = FTP(ftp_data['server'])
    ftp.login(ftp_data['username'], ftp_data['password'])
    ftp.cwd(path)
    if extension == '.pkl':
        _save_obj(object, 'externals/tmp/tmp_file')
        fp = open('externals/tmp/tmp_file' + extension, 'rb')
    else:
        fp = open(object, 'rb')
    ftp.storbinary('STOR %s' % remotefile, fp)
    ftp.quit()


def _ftp_download(ftp_data, path, filename, extension='.pkl'):
    ftp = FTP(ftp_data['server'])
    ftp.login(ftp_data['username'], ftp_data['password'])
    ftp.cwd(path)

    if extension == '.pkl':
        handle = open('externals/tmp/tmp_file' + extension, 'wb')
        ftp.retrbinary('RETR %s' % filename + extension, handle.write, 8192)
        ftp_file = _load_obj('externals/tmp/tmp_file')
    else:
        handle = open('externals/tmp/' + filename, 'wb')
        ftp.retrbinary('RETR %s' % filename, handle.write, 8192)
        ftp_file = open('externals/tmp/'+filename, 'rb')
    ftp.quit()
    return ftp_file


def _sftp_upload(sftp_data, path, object, filename, extension='.pkl'):
    sftp = pysftp.Connection(sftp_data['server'], username=sftp_data['username'], password=sftp_data['password'])
    sftp.cwd(path)
    if extension == '.pkl':
        _save_obj(object, 'externals/tmp/tmp_file')
        fp = 'externals/tmp/tmp_file' + extension
        sftp.put(localpath=fp, remotepath=filename + extension, preserve_mtime=True)
    else:
        fp = filename
        remote_fp = filename.split('/')[-1]
        sftp.put(localpath=fp, remotepath=remote_fp, preserve_mtime=True)

    sftp.close()


def _sftp_download(sftp_data, path, filename, extension='.pkl'):
    sftp = pysftp.Connection(sftp_data['server'], username=sftp_data['username'], password=sftp_data['password'])
    sftp.cwd(path)
    if extension == '.pkl':
        local_name = 'externals/tmp/tmp_file' + extension
        sftp.get(filename + extension, local_name, preserve_mtime=True)
        sftp_file = _load_obj('externals/tmp/tmp_file')
    else:
        sftp.get(filename, 'externals/tmp/' + filename, preserve_mtime=True)
        sftp_file = open('externals/tmp/' + filename)
    sftp.close()
    return sftp_file


def _parse_data(input_report, remote_server, parsing_criteria):
    parsed_data = {'report': "", 'image_files': [], 'rois': [],'abbvs':[], 'ne_terms': []}
    image_list=[]
    if remote_server['server_type'] == 'ftp':
        ftp = FTP(remote_server['server'])
        ftp.login(remote_server['username'], remote_server['password'])
        ftp.cwd(parsing_criteria['image_folder'])
        image_list = ftp.nlst()
        ftp.quit()
    elif remote_server['server_type'] == 'sftp':
        sftp = pysftp.Connection(remote_server['server'],
                                 username=remote_server['username'],
                                 password=remote_server['password'])
        sftp.cwd(parsing_criteria['image_folder'])
        image_list = sftp.listdir()
        sftp.close()
    if parsing_criteria['format'] == "xml":
        tree = ET.parse(input_report)
        root = tree.getroot()
        new_report = ""
        for element in root.iter(parsing_criteria['report_headers']):
            if parsing_criteria['report_label']:
                new_report += "###" + element.attrib[parsing_criteria['report_label']] + "\n"
            if element.text:
                new_report += element.text + "\n"
        parsed_data['report'] = new_report
        new_ner = []
        if parsing_criteria['NE_headers']:
            for i in range(len(parsing_criteria['NE_headers'])):
                for element in root.findall(parsing_criteria['NE_headers'][i]):
                    els = element.text.rstrip().split('/')[0]
                    els = els.split(',')
                    new_ner += els
            parsed_data['ne_terms'] = list(set(new_ner))
        new_abbvs = []
        if parsing_criteria['abbv_headers']:
            for i in range(len(parsing_criteria['abbv_headers'])):
                for element in root.findall(parsing_criteria['abbv_headers'][i]):
                    els = element.text.rstrip().split('/')[0]
                    els = els.split(',')
                    new_ner += els
            parsed_data['abbvs'] = list(set(new_abbvs))
        new_image = []
        new_rois=[]
        if parsing_criteria['image_header']:
            for element in root.iter(parsing_criteria['image_header']):
                if parsing_criteria['roi_header']:
                    rois={}
                    for element in root.iter(parsing_criteria['roi_header']):
                            rois[element.attrib[parsing_criteria['roi_coordinate']]]=int(element.text)
                    new_rois.append(list(rois.values()))
                for f in image_list:
                    if element.attrib[parsing_criteria['image_label']] in f:
                        new_image.append(f)
                        break
            parsed_data['image_files'] = new_image

    if parsing_criteria['format'] == 'plain':
        new_report = ""
        report = open(input_report).read().rstrip().lstrip().split('\n')
        if {'FINDINGS', 'COMPARISON', 'INDICATION', 'IMPRESSION'} & set(report):
            report_to_dict = internal_functions.section_string_to_dict(report)
            for k, v in report_to_dict.items():
                new_report += '###' + k + '\n'
                new_report += v + '\n'
        else:
            new_report = '\n'.join(report)
        parsed_data['report'] = new_report
        report_id = input_report.split('/')[-1].split('''\\''')[-1].split('_')[0]
        new_image = []
        for f in image_list:
            if f.startswith(report_id):
                new_image.append(f)
        parsed_data['image_files'] = new_image

    return parsed_data


def _create_case_df(case_path, remote_server=False):
    if remote_server:
        df = pd.DataFrame(columns=['Case_ID', 'Case_Location', 'Original_Location',
                                   'Image_Features', 'Doc_Embedding', 'NE_Detected',
                                   'Abbrv_#', 'Validation_Status', 'First_In', 'Last_Modified', 'Modified_By'])
    else:
        df = pd.DataFrame(columns=['Case_ID', 'Case_Location', 'Original_Location',
                                   'Image_Features', 'Doc_Embedding', 'NE_Detected',
                                   'Abbrv_#', 'Validation_Status', 'First_In', 'Last_Modified'])

    pd.to_pickle(df, case_path, compression='zip')


def _sync_case_df(server_data, local_case_path, remote_case_path):
    remote_path = '/'.join(remote_case_path.split('/')[:-1])
    if remote_path.endswith('.zip'):
        remote_path = ""
    remote_file = remote_case_path.split('/')[-1]
    if server_data['server_type'] == 'ftp':
        _ftp_upload(server_data, remote_path, local_case_path, remote_file, extension='.zip')
    elif server_data['server_type'] == 'sftp':
        _sftp_upload(server_data, remote_path, None, local_case_path, extension='.zip')


# Que formatos de datos vamos a considerar? XML y...?
class Storage_Unit:
    # Basic functions
    def __init__(self):
        self.case_host = ""
        self.originals_host = ""
        self.case_prefix = ""
        self.data_format = ""
        self.case_index_path = ""
        self.remote_server = None
        self.credentials = ["",""]
        self.server_type = ""
        self.real_case_index = ""

    def set_remote_server(self, server_type, remote_server, username, password):
        self.server_type = server_type
        self.remote_server = remote_server
        self.credentials = [username, password]

    def set_storage_unit(self, case_host, originals_host, case_prefix, case_index_path):
        self.case_host = case_host
        self.case_prefix = case_prefix
        self.originals_host = originals_host
        # self.case_index_path=case_index_path
        if case_index_path.endswith('.zip'):
            # If the file exists, we make a local copy, which is uploaded after each operation
            if self.remote_server:
                self.remote_case_index = case_index_path
                case_folder = '/'.join(case_index_path.split('/')[:-1])
                case_file = case_index_path.split('/')[-1]
                if self.get_remote_server()['server_type'] == 'ftp':
                    index_file = _ftp_download(self.get_remote_server(), case_folder, case_file, extension='.zip')
                elif self.get_remote_server()['server_type'] == 'sftp':
                    index_file = _sftp_download(self.get_remote_server(), case_folder, case_file, extension='.zip')
                index_file_name = index_file.name
                index_file.close()
                shutil.copy2(index_file_name, 'externals/' + case_file)
                self.case_index_path = 'externals/' + case_file
            else:
                self.case_index_path = 'externals/' + case_index_path
        else:
            _create_case_df('externals/' + case_index_path, True)
            self.case_index_path = 'externals/' + case_index_path+'.zip'
            self.remote_case_index = self.originals_host + '/' + case_index_path

    def change_case_host(self, new_case_host):
        self.case_host = new_case_host

    def change_csv_path(self, new_case_index_path):
        self.case_index_path = new_case_index_path

    def get_storage_unit(self):
        return self.case_host, self.originals_host, self.case_prefix, self.case_index_path

    def get_originals_host(self):
        return self.originals_host

    def get_case_host(self):
        return self.case_host

    def get_remote_server(self):
        return {'server_type': self.server_type, 'server': self.remote_server,
                'username': self.credentials[0], 'password': self.credentials[1]}

    def get_csv_path(self):
        return self.case_index_path

    def get_case_prefix(self):
        return self.case_prefix

    def get_remote_case_index(self):
        return self.remote_case_index

    ###Class functions

    def dump_case(self, new_case):
        """Stores a created case in the system and adds its corresponding entry into the case index
        INPUT: Case to dump"""
        df = pd.read_pickle(self.get_csv_path())
        if self.remote_server:
            if self.get_remote_server()['server_type'] == 'ftp':
                _ftp_upload(self.get_remote_server(), self.get_case_host(), new_case,
                            self.get_case_prefix() + '-' + str(len(df)))
            elif self.get_remote_server()['server_type'] == 'sftp':
                _sftp_upload(self.get_remote_server(), self.get_case_host(), new_case,
                             self.get_case_prefix() + '-' + str(len(df)))
        else:
            _save_obj(new_case, self.get_case_host() + '/' + self.get_case_prefix() + '-' + str(len(df)))
        im_vectors = []
        if new_case.get_problem().get_image_file():
            for i in new_case.get_problem().get_image_file():
                im_vectors.append(internal_functions.image_feature_extraction(i))
                remote_name = i.split('/')[-1]
                if self.get_remote_server()['server_type'] == 'ftp':
                    _ftp_upload(self.get_remote_server(), self.get_originals_host() + '/images/', i, remote_name,
                                extension='image')
                elif self.get_remote_server()['server_type'] == 'sftp':
                    _sftp_upload(self.get_remote_server(), self.get_originals_host() + '/images/', None, i,
                                 extension='image')
                new_case.get_problem().set_image_files([os.path.join(self.originals_host + '/images/', remote_name)])
                os.remove(i)
        abbs = internal_functions.get_abbr_ratio(new_case.get_solution().get_section_report())
        if self.remote_server:
            new_row = [self.get_case_prefix() + '-' + str(len(df)),
                       self.get_case_host() + '/' + self.get_case_prefix() + '-' + str(len(df)),
                       self.get_originals_host(), im_vectors,
                       internal_functions.get_document_embedding(new_case.get_solution().get_section_report()),
                       new_case.get_problem().get_term_list(), abbs, 'Pending',
                       datetime.datetime.utcnow(), datetime.datetime.utcnow(), self.get_remote_server()['username']]
        else:
            new_row = [self.get_case_prefix() + '-' + str(len(df)),
                       self.get_case_host() + '/' + self.get_case_prefix() + '-' + str(len(df)),
                       self.get_originals_host(), im_vectors,
                       internal_functions.get_document_embedding(new_case.get_solution().get_section_report()),
                       new_case.get_problem().get_term_list(), abbs, 'Pending',
                       datetime.datetime.utcnow(), datetime.datetime.utcnow()]
        df.loc[len(df)] = new_row
        df.to_pickle(self.get_csv_path())
        if self.remote_server:
            _sync_case_df(self.get_remote_server(), self.get_csv_path(), self.get_remote_case_index())

    def create_new_case(self, new_case, corrected_report=""):
        """Creates and stores a new case.
        If no manual solution is provided, it will be inferred automatically from the system and marked as 'Pending'
        INPUT: Input report [Image_files, Solution files]"""
        if corrected_report:
            new_sections = corrected_report
        else:
            new_sections = internal_functions.section_text(new_case.get_problem().get_report())
        new_abbvs = internal_functions.disambiguate_abbreviation(new_case.get_solution().get_section_report())
        new_ner = internal_functions.get_entities(new_case.get_problem().get_report(),
                                                  new_case.get_problem().get_term_list())
        new_case.get_problem().set_term_list(new_ner)
        new_case.set_solution(new_sections=new_sections, new_abbvs=new_abbvs)
        im_vectors = []
        if new_case.get_problem().get_image_file():
            for i in new_case.get_problem().get_image_file():
                im_vectors.append(internal_functions.image_feature_extraction(i))
                remote_name = i.split('/')[-1]
                if self.get_remote_server()['server_type'] == 'ftp':
                    _ftp_upload(self.get_remote_server(), self.get_originals_host() + '/images/', i, remote_name,
                                extension='image')
                elif self.get_remote_server()['server_type'] == 'sftp':
                    _sftp_upload(self.get_remote_server(), self.get_originals_host() + '/images/', None, i,
                                 extension='image')
                # new_case.get_problem().set_image_files([os.path.join(self.originals_host + '/images/', remote_name)])
                # os.remove(i)
        abbs = internal_functions.get_abbr_ratio(new_case.get_problem().get_report(),
                                                 new_case.get_problem().get_abbrs())
        similar_cases = self.find_top_cases(new_case, {'n': 3})
        new_case.get_solution().set_related_cases(similar_cases)
        df = pd.read_pickle(self.get_csv_path())
        if self.remote_server:
            if self.get_remote_server()['server_type'] == 'ftp':
                _ftp_upload(self.get_remote_server(), self.get_case_host(), new_case,
                            self.get_case_prefix() + '-' + str(len(df)))
            elif self.get_remote_server()['server_type'] == 'sftp':
                _sftp_upload(self.get_remote_server(), self.get_case_host(), new_case,
                             self.get_case_prefix() + '-' + str(len(df)))
        else:
            _save_obj(new_case, self.get_case_host() + '/' + self.get_case_prefix() + '-' + str(len(df)))

        if self.remote_server:
            new_row = [self.get_case_prefix() + '-' + str(len(df)),
                       self.get_case_host() + '/' + self.get_case_prefix() + '-' + str(len(df)),
                       self.get_originals_host(), im_vectors,
                       internal_functions.get_document_embedding(new_case.get_solution().get_section_report()),
                       new_ner, abbs, 'Pending',
                       datetime.datetime.utcnow(), datetime.datetime.utcnow(), self.get_remote_server()['username']]
        else:
            new_row = [self.get_case_prefix() + '-' + str(len(df)),
                       self.get_case_host() + '/' + self.get_case_prefix() + '-' + str(len(df)),
                       self.get_originals_host(), im_vectors,
                       internal_functions.get_document_embedding(new_case.get_solution().get_section_report()),
                       new_ner, abbs, 'Pending',
                       datetime.datetime.utcnow(), datetime.datetime.utcnow()]
        df.loc[len(df)] = new_row
        df.to_pickle(self.get_csv_path())
        if self.remote_server:
            _sync_case_df(self.get_remote_server(), self.get_csv_path(), self.get_remote_case_index())

    def create_case_set(self, parsing_criteria, report_list=''):
        """Generates a set of cases from existing reports
        INPUT: Parsing criteria [Report_list]"""
        if self.remote_server:
            if self.get_remote_server()['server_type'] == 'ftp':
                ftp = FTP(self.get_remote_server()['server'])
                ftp.login(self.get_remote_server()['username'], self.get_remote_server()['password'])
                ftp.cwd(self.get_originals_host() + '/reports/')
                report_list = ftp.nlst()
                ftp.quit()
            elif self.get_remote_server()['server_type'] == 'sftp':
                sftp = pysftp.Connection(self.get_remote_server()['server'],
                                         username=self.get_remote_server()['username'],
                                         password=self.get_remote_server()['password'])
                sftp.cwd(self.get_originals_host() + '/reports/')
                report_list = sftp.listdir()
                sftp.close()
        for report in report_list:
            if self.get_remote_server()['server_type'] == 'ftp':
                tmp_report = _ftp_download(self.get_remote_server(), self.get_originals_host() + '/reports/', report,
                                           extension='.txt')
            elif self.get_remote_server()['server_type'] == 'sftp':
                tmp_report = _sftp_download(self.get_remote_server(), self.get_originals_host() + '/reports/', report,
                                            extension='.txt')
            tmp_report.close()
            parsed_data = _parse_data('externals/tmp/' + report, self.get_remote_server(), parsing_criteria)
            os.remove('externals/tmp/'+report)
            # Returning the sectioned reports, if there are, and the name of the image files
            new_case = internal_functions.Case()
            new_case.set_problem(new_report=parsed_data['report'],new_image=parsed_data['image_files'])
            report_sectioned = parsed_data['report'].rstrip().lstrip().split('\n')
            solution = ""
            if bool({'###FINDINGS', '###COMPARISON', '###INDICATION', '###IMPRESSION'} & set(report_sectioned)):
                solution = parsed_data['report']
            self.create_new_case(new_case, solution)

    def recover_single_case(self, case_ID):
        """Recovers a single case given its ID
        INPUT: Case_ID
        OUTPUT: The case object"""
        if self.remote_server:
            if self.get_remote_server()['server_type'] == 'ftp':
                return _ftp_download(self.get_remote_server(), self.get_case_host(), case_ID)
            elif self.get_remote_server()['server_type'] == 'sftp':
                return _sftp_download(self.get_remote_server(), self.get_case_host(), case_ID)

        else:
            return _load_obj(self.case_host + '/' + case_ID)

    def get_case_entities(self, case_ID):
        """Retrieves the named entities detected within a case
        INPUT: Case_ID
        OUTPUT: Dictionary containing each detected entity and its type"""
        df = pd.read_pickle(self.case_index_path)
        return df.loc[df['Case_ID'] == case_ID, 'NE_Detected'].to_dict()

    def change_case_solution(self, case_ID, new_report, change_validation=''):
        """Changes the stored solution of a given case.
        INPUT: Case_ID of the case to change, New Case Solution and, optionally, a new validation value (this should only be changed by an expert)"""
        case = self.recover_single_case(case_ID)
        case.get_solution().set_section_report(new_report)
        if self.remote_server:
            if self.get_remote_server()['server_type'] == 'ftp':
                _ftp_upload(self.get_remote_server(), self.get_case_host(), case, case_ID)
            elif self.get_remote_server()['server_type'] == 'sftp':
                _sftp_upload(self.get_remote_serparsver(), self.get_case_host(), case, case_ID)
        else:
            _save_obj(case, self.get_case_host() + '/' + case_ID)
        df = pd.read_pickle(self.case_index_path)
        df.loc[df['Case_ID'] == case_ID, 'Last_Modified'] = datetime.datetime.utcnow()
        if change_validation:
            df.loc[df['Case_ID'] == case_ID, 'Validation_Status'] = change_validation
        if self.remote_server:
            df.loc[df['Case_ID'] == case_ID, 'Modified_By'] = self.get_remote_server()['username']
        df.to_pickle(self.case_index_path)
        if self.remote_server:
            _sync_case_df(self.get_remote_server(), self.get_csv_path(), self.get_remote_case_index())

    def update_related_entities(self, case_ID, new_related):
        """Updates the stored related entities of a given case
        INPUT: Case_ID, List of related cases"""
        case = self.recover_single_case(case_ID)
        case.get_solution().set_suggested_ner(new_related)
        if self.remote_server:
            if self.get_remote_server()['server_type'] == 'ftp':
                _ftp_upload(self.get_remote_server(), self.get_case_host(), case, case_ID)
            elif self.get_remote_server()['server_type'] == 'sftp':
                _sftp_upload(self.get_remote_server(), self.get_case_host(), case, case_ID)
        else:
            _save_obj(case, self.get_case_host() + '/' + case_ID)
        df = pd.read_pickle(self.case_index_path)
        df.loc[df['Case_ID'] == case_ID, 'Last_Modified'] = datetime.datetime.utcnow()
        if self.remote_server:
            df.loc[df['Case_ID'] == case_ID, 'Modified_By'] = self.get_remote_server()['username']
        df.to_pickle(self.case_index_path)
        if self.remote_server:
            _sync_case_df(self.get_remote_server(), self.get_csv_path(), self.get_remote_case_index())

    def update_related_cases(self, case_ID, new_related):
        """Updates the stored related cases of a given case
        INPUT: Case_ID, List of related cases"""
        case = self.recover_single_case(case_ID)
        case.get_solution().set_related_cases(new_related)
        if self.remote_server:
            if self.get_remote_server()['server_type'] == 'ftp':
                _ftp_upload(self.get_remote_server(), self.get_case_host(), case, case_ID)
            elif self.get_remote_server()['server_type'] == 'sftp':
                _sftp_upload(self.get_remote_server(), self.get_case_host(), case, case_ID)
        else:
            _save_obj(case, self.get_case_host() + '/' + case_ID)
        df = pd.read_pickle(self.case_index_path)
        df.loc[df['Case_ID'] == case_ID, 'Last_Modified'] = datetime.datetime.utcnow()
        if self.remote_server:
            df.loc[df['Case_ID'] == case_ID, 'Modified_By'] = self.get_remote_server()['username']
        df.to_pickle(self.case_index_path)
        if self.remote_server:
            _sync_case_df(self.get_remote_server(), self.get_csv_path(), self.get_remote_case_index())

    def link_similar_n_cases(self):
        df = pd.read_pickle(self.case_index_path)
        for i, row in df.iterrows():
            i1 = row['Image_Features']
            matches_i1 = {}
            for j, j_row in df.iterrows():
                cum = []
                for (i_case, i_potential) in itertools.product(i1, j_row['Image_Features']):
                    cum.append(1 - spatial.distance.cosine(i_case, i_potential))
                if cum != []:
                    matches_i1[j] = np.mean(cum)
                else:
                    matches_i1[j] = 1
            i2 = row['Doc_Embedding']
            matches_i2 = {j: 1 - spatial.distance.cosine(i2, j_row['Doc_Embedding']) for j, j_row in df.iterrows()}
            i3 = row['NE_Detected']
            matches_i3 = {j: list(set(i3) & set(j_row['NE_Detected'])) for j, j_row in df.iterrows()}
            matches_total = {j_row['Case_ID']: np.mean([matches_i1[j], matches_i2[j], len(matches_i3[j])]) for j, j_row
                             in df.iterrows()}
            sorted_d = sorted(matches_total.items(), key=operator.itemgetter(1), reverse=True)
            cases_to_input = sorted_d[1:6]
            cases_to_input = [a_tuple[0] for a_tuple in cases_to_input]
            self.update_related_cases(row['Case_ID'], cases_to_input)
            new_entities = internal_functions.get_case_related_entities(self, cases_to_input)
            self.update_related_entities(row['Case_ID'], new_entities)

    def find_top_cases(self, case, input_criteria=None):
        """Finds the most similar N cases to a given case according to the provided criteria
        INPUT: Present case, [Search criteria (If no input criteria is provided, the default values are used)]
        OUTPUT: Dictionary with the retrieved cases and their metrics with respect to the present case"""
        if input_criteria is None:
            criteria = {'i1': -1, 'i2': -1, 'i3': [], 'i4': -1, 'n': -1, 'query_type': 'or', 'discarded_cases': []}
        else:
            default_criteria = {'i1': -1, 'i2': -1, 'i3': [], 'i4': -1, 'n': -1, 'query_type': 'or',
                                'discarded_cases': []}
            criteria = default_criteria.copy()
            criteria.update(input_criteria)

        i1_value = case.get_problem().get_image_file()
        roi_value = case.get_problem().get_roi_coordinates()
        i2_value = case.get_problem().get_report()
        i3_value = case.get_problem().get_term_list()
        case_im_embeddings = []
        for i in i1_value:
            case_im_embeddings.append(internal_functions.image_feature_extraction(i, roi=roi_value))
        case_doc_embedding = internal_functions.get_document_embedding(i2_value)
        case_terms = internal_functions.get_entities(i2_value, i3_value)
        if not case_terms:
            case_terms = {"Empty": "empty"}
        df = pd.read_pickle(self.case_index_path)
        potential_cases = []
        if not criteria['discarded_cases']:
            case_list = df
        else:
            case_list = df[(~df.Case_ID.isin(criteria['discarded_cases']))]
        for i, row in case_list.iterrows():
            if row['Validation_Status'] == 'Rejected':
                continue
            metrics = {'i1': 1, 'i2': 1, 'i3': [], 'i4': 1}
            i1_case = row['Image_Features']
            i2_case = row['Doc_Embedding']
            i3_case = row['NE_Detected']
            i4_case = row['Abbrv_#']
            # First, we compare the images
            if i1_value:
                metrics['i1'] = np.mean([1 - spatial.distance.cosine(i_case, i_potential) for (i_case, i_potential) in
                                         itertools.product(i1_case, case_im_embeddings)])
            metrics['i2'] = 1 - spatial.distance.cosine(np.array(case_doc_embedding), np.array(i2_case))
            metrics['i3'] = list(set(i3_case.keys()) & set(case_terms.keys()))
            if not metrics['i3']:
                metrics['i3'] = [""]
            metrics['i4'] = i4_case
            # Then, we check if the conditions are joint or disjoint
            if criteria['query_type'] == 'and':
                if metrics['i1'] >= criteria['i1'] and metrics['i2'] >= criteria['i2'] and \
                        len(metrics['i3']) >= len(criteria['i3']) and metrics['i4'] >= criteria['i4']:
                    metrics['total'] = np.mean(
                        [np.mean(metrics['i1']), metrics['i2'], float(len(metrics['i3']) / len(case_terms.keys())),
                         metrics['i4']])
                    metrics['Case_ID'] = row['Case_ID']
                    potential_cases.append(metrics)

            elif criteria['query_type'] == 'or':
                if metrics['i1'] >= criteria['i1'] or metrics['i2'] >= criteria['i2'] or \
                        metrics['i3'] or metrics['i4'] >= criteria['i4']:
                    metrics['total'] = np.mean(
                        [np.mean(metrics['i1']), metrics['i2'], float(len(metrics['i3']) / len(case_terms.keys())),
                         metrics['i4']])
                    metrics['Case_ID'] = row['Case_ID']
                    potential_cases.append(metrics)

        if criteria['n'] != -1 and criteria['n'] <= len(potential_cases):
            newlist = sorted(potential_cases, key=operator.itemgetter('total'), reverse=True)
            potential_cases = newlist[:criteria['n']]

        return potential_cases

    def download_image(self,image_file):
        """Downloads the required image file from the server, if there is, and stores it into the tmp folder
        INPUT: Name of the image file to download."""
        if self.get_remote_server()['server_type'] == 'ftp':
            _ftp_download(self.get_remote_server(), self.get_originals_host()+'/images/'
                                       , image_file, extension='image')
        elif self.get_remote_server()['server_type'] == 'sftp':
            _sftp_download(self.get_remote_server(), self.get_originals_host()+'/images/',
                                        image_file, extension='image')





import internal_functions as it
from flask import Flask, request, render_template, redirect, flash, url_for, session
from flask_wtf import FlaskForm
from wtforms import StringField, FileField, validators, PasswordField
import os
from werkzeug.utils import secure_filename
import shutil
import ftplib
from ftplib import FTP
import pandas as pd
import pysftp
import pickle
import pydicom
import cv2
from pydicom.pixel_data_handlers.util import apply_color_lut

app = Flask(__name__)
app.config.from_object(__name__)
key = str(os.urandom(16).hex())
app.config['SECRET_KEY'] = key
app.config["UPLOAD_FOLDER"] = 'externals/tmp/'
app.config["STATIC_FOLDER"] = 'static/'
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "DCM"]

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'dcm'])

storage_unit = it.Storage_Unit()
set_storage_unit = False
current_problem = None
server = False
discarded_cases = []
scored_cases=[]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


class STUForm(FlaskForm):
    case_folder = StringField('case_folder', validators=[
        validators.DataRequired()])
    originals_folder = StringField('originals_folder', validators=[
        validators.DataRequired()])
    username = StringField('username', validators=[validators.DataRequired()])
    password = PasswordField('user_pass', validators=[validators.DataRequired()])
    case_prefix = StringField('case_prefix', validators=[validators.DataRequired()])
    case_index = StringField('case_index', validators=[validators.DataRequired()])


class CaseForm(FlaskForm):
    input_report = StringField('input_report', validators=[validators.DataRequired()])
    image_file = FileField('image_file')
    terms = StringField('ne_terms')
    abbrvs = StringField('abbrvs')


@app.context_processor
def utilities():
    def get_existing_cases():
        df = pd.read_pickle(storage_unit.get_csv_path())
        existing_case_data = {}
        for i, row in df.iterrows():
            existing_case_data[row['Case_ID']] = [row['Case_Location'], row['NE_Detected'], row['Abbrv_#'],
                                                  row['Validation_Status'], row['First_In'], row['Last_Modified']]
            if server:
                existing_case_data[row['Case_ID']].append(row['Modified_By'])
        return existing_case_data

    def get_cases_to_validate():
        df = pd.read_pickle(storage_unit.get_csv_path())
        cases_to_validate = {}
        for i, row in df.iterrows():
            if row['Validation_Status'] == 'Pending':
                case_content = storage_unit.recover_single_case(row['Case_ID'])
                problem = {'report': case_content.get_problem().get_report(),
                           'terms': case_content.get_problem().get_term_list(),
                           'abbvs': case_content.get_problem().get_abbrs()}
                solution = {'report': it.section_string_to_dict(
                    case_content.get_solution().get_section_report().rstrip().split('\n')),
                    'terms': case_content.get_solution().get_sugg_ner(),
                    'abbvs': case_content.get_solution().get_sugg_abbvs(),
                    'related_cases': case_content.get_solution().get_related_cases()
                }
                cases_to_validate[row['Case_ID']] = {'problem': problem,
                                                     'solution': solution}
        return cases_to_validate

    def get_section_model():
        if os.path.exists('externals/cpu_section_model.pkl'):
            section_model = _load_obj('externals/cpu_section_model')
            accuracy = section_model['acc']
            timestamp = section_model['timestamp']
            del section_model
        elif os.path.exists('externals/cuda_section_model.pkl'):
            section_model = _load_obj('externals/cuda_section_model')
            accuracy = section_model['acc']
            timestamp = section_model['timestamp']
            del section_model
        else:
            accuracy = '0'
            timestamp = 'None'
        return {'acc': accuracy, 'time': timestamp}

    def get_scoring_model():
        if os.path.exists('externals/scoring_model.pkl'):
            scoring_model = _load_obj('externals/scoring_model')
            accuracy = scoring_model['acc']
            timestamp = scoring_model['timestamp']
            del scoring_model
        else:
            accuracy = '0'
            timestamp = 'None'
        return {'acc': accuracy, 'time': timestamp}

    return dict(get_existing_cases=get_existing_cases, get_cases_to_validate=get_cases_to_validate,
                get_section_model=get_section_model, get_scoring_model=get_scoring_model)


@app.route("/", methods=['GET', 'POST'])
def init():
    # We clear the temporary data stored in the cache
    filelist = [f for f in os.listdir(app.config['STATIC_FOLDER']) if allowed_file(f)]
    for f in filelist:
        os.remove(os.path.join(app.config['STATIC_FOLDER'], f))
    filelist = [f for f in os.listdir(app.config['UPLOAD_FOLDER'])]
    for f in filelist:
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))
    global current_problem
    current_problem = None
    global set_storage_unit
    if not set_storage_unit:
        form = STUForm()
        return render_template('createSTU.html', form=form)
    else:
        form = CaseForm()
        return render_template('init.html', form=form)


@app.route("/exit_stu")
def exit_stu():
    global set_storage_unit
    set_storage_unit=False
    global server
    server=False
    global storage_unit
    storage_unit = it.Storage_Unit()
    return redirect(url_for('init'))


@app.route("/set_stu", methods=['POST'])
def set_stu():
    global storage_unit
    global set_storage_unit
    global server
    if request.form['stu_option'] == 'ecgen':
        # def set_storage_unit(self, case_host, originals_host, data_format, case_prefix, case_index_path)
        storage_unit.set_storage_unit('externals/ecgen-samples/case_set/', 'externals/ecgen-samples/',
                                      'ecgen-case', 'ecgen-samples/ecgen_case_index.zip')
        set_storage_unit = True
        server=False
    else:
        form = STUForm()
        form.validate()
        server_name = request.form['server_name']
        username = request.form['username']
        password = request.form['user_pass']
        server_type = request.form['server_type']
        if server_type == 'ftp':
            try:
                ftp = FTP(server_name)
                ftp.login(username, password)
                ftp.quit()
            except ftplib.all_errors as e:
                flash(str(e))
                return redirect(url_for('init'))
        elif server_type == 'sftp':
            try:
                sftp = pysftp.Connection(server_name, username=username, password=password)
                sftp.close()
            except Exception as e:
                flash(str(e))
                return redirect(url_for('init'))
        server = True
        storage_unit.set_remote_server(server_type, server_name, username, password)
        storage_unit.set_storage_unit(request.form['case_folder'], request.form['original_data'],
                                      request.form['case_prefix'], request.form['case_index_file'])
        set_storage_unit = True
    session['server']=server
    return redirect(url_for('init'))


@app.route("/create_problem", methods=['POST'])
def create_problem():
    form = CaseForm(request.form)
    global discarded_cases
    discarded_cases = []
    global scored_cases
    scored_cases = []
    if form.validate_on_submit():
        data = {'report': request.form['input_report'], 'ne_terms': request.form['ne_terms'].split(','),
                'abbrvs': request.form['abbrvs'].split(',')}
        checked = 'rois' in request.form
        session['rois'] = checked
        session['update_cases'] = False
        image = request.files["image_file"]
        if image and allowed_file(image.filename):
            data['images'] = [os.path.join(app.config['UPLOAD_FOLDER'], image.filename)]
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config['STATIC_FOLDER'], filename))
            image_src = os.path.join(app.config['STATIC_FOLDER'], filename)
            if filename.endswith('.dcm') or filename.endswith('.DCM'):
                dicom_image = pydicom.dcmread(image_src)
                arr = dicom_image.pixel_array
                rgb_image = apply_color_lut(arr, dicom_image, palette='PET')
                gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
                cv2.imwrite(app.config['STATIC_FOLDER'] + filename.replace('.dcm', '.png'), gray)
                image_src = os.path.join(app.config['STATIC_FOLDER'], filename.replace('.dcm', '.png'))
            shutil.copy(os.path.join(app.config['STATIC_FOLDER'], filename),
                        os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            data['images'] = []
            image_src = ""

        global current_problem
        current_problem = it.init_cbr(data)
        ners = it.get_entities(current_problem.get_problem().get_report(),
                               current_problem.get_problem().get_term_list())
        ners={n.replace('xxxx',''):v for n,v in ners.items()}
        current_problem.get_problem().set_term_list(ners)
    return render_template('cbr_retrieve.html', content=current_problem.get_problem().get_report(),
                           image_src=image_src, ners=ners)


@app.route("/cbr_retrieve", methods=['POST'])
def cbr_retrieve():
    criteria = {}
    global current_problem
    if session['rois']:
        section_x = request.form['roi_selected_x']
        section_y = request.form['roi_selected_y']
        section_h = request.form['roi_selected_h']
        section_w = request.form['roi_selected_w']
        if section_x == "" or section_y == "" or section_h == "" or section_w == "":
            flash("ERROR")
            filename = current_problem.get_problem().get_image_file()[0].split('/')[-1]
            image_src = os.path.join(app.config['STATIC_FOLDER'], filename)
            ners = current_problem.get_problem().get_term_list()
            ners={n.replace('xxxx',''):v for n,v in ners.items()}
            return render_template('cbr_retrieve.html', content=current_problem.get_problem().get_report(),
                                   image_src=image_src, ners=ners)
        else:
            current_problem.get_problem().set_rois([int(section_x), int(section_y), int(section_w), int(section_h)])
    ners = request.form.getlist('ners')
    criteria['i3'] = ners
    i2_value = request.form['i2_value_name']
    criteria['i2'] = float(i2_value) / 100
    if current_problem.get_problem().get_image_file():
        i1_value = request.form['i1_value_name']
        criteria['i1'] = float(i1_value) / 100
    i4_value = request.form['i4_value_name']
    criteria['i4'] = float(i4_value) / 100
    if 'condition' in request.form:
        criteria['query_type'] = 'and'
    else:
        criteria['query_type'] = 'or'
    n_cases = request.form.get('n_cases')
    if n_cases != 'All':
        criteria['n'] = int(n_cases)
    if session['update_cases']:
        global discarded_cases
        discarded_cases += [case for case in current_problem.get_solution().get_related_cases()]
        criteria['discarded_cases'] = discarded_cases
    global scored_cases
    scored_cases = it.cbr_retrieval(storage_unit, current_problem, criteria)
    current_problem, score = it.cbr_reuse(storage_unit, current_problem, scored_cases)
    sectioned_report = current_problem.get_solution().get_section_report().rstrip().split('\n')
    sectioned_report_to_dict = it.section_string_to_dict(sectioned_report)
    return render_template('cbr_reuse.html', previous_report=current_problem.get_problem().get_report(),
                           sectioned_report=sectioned_report_to_dict,
                           sugg_problems=current_problem.get_solution().get_sugg_ner()['problem'],
                           sugg_treatments=current_problem.get_solution().get_sugg_ner()['treatment'],
                           sugg_tests=current_problem.get_solution().get_sugg_ner()['test'],
                           sugg_abbrv=current_problem.get_solution().get_sugg_abbvs(),
                           related_cases=scored_cases, score=score)


@app.route("/cbr_reuse", methods=['POST'])
def cbr_reuse():
    global current_problem
    choice = request.form.get('submit_case')
    sectioned_report = current_problem.get_solution().get_section_report().rstrip().split('\n')
    sectioned_report_to_dict = it.section_string_to_dict(sectioned_report)
    st_name = storage_unit.get_case_host()
    if choice == "yes":
        if st_name != 'externals/ecgen-samples/case_set/' and st_name != 'externals/mimic-cxr-samples/case_set/':
            storage_unit.dump_case(current_problem)
            storage_unit.link_similar_n_cases()
        score = it.score_case('\n'.join(sectioned_report))
        # Clear current case so that is empty
        # Remove data in static
        # if current_problem.get_problem().get_image_file():
        #     filename = current_problem.get_problem().get_image_file()[0].split('/')[-1]
        #     os.remove(os.path.join(app.config['STATIC_FOLDER'], filename))
        flash("Success!")
        return render_template('cbr_reuse.html', previous_report=current_problem.get_problem().get_report(),
                               sectioned_report=sectioned_report_to_dict,
                               sugg_problems=current_problem.get_solution().get_sugg_ner()['problem'],
                               sugg_treatments=current_problem.get_solution().get_sugg_ner()['treatment'],
                               sugg_tests=current_problem.get_solution().get_sugg_ner()['test'],
                               sugg_abbrv=current_problem.get_solution().get_sugg_abbvs(),
                               related_cases=scored_cases,score=score)
    elif choice == "retry":
        if current_problem.get_problem().get_image_file():
            filename = current_problem.get_problem().get_image_file()[0].split('/')[-1]
            # os.remove(os.path.join(app.config['STATIC_FOLDER'], filename))
            image_src = os.path.join(app.config['STATIC_FOLDER'], filename)
            if filename.endswith('.dcm') or filename.endswith('.DCM'):
                dicom_image = pydicom.dcmread(image_src)
                arr = dicom_image.pixel_array
                rgb_image = apply_color_lut(arr, dicom_image, palette='PET')
                gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
                cv2.imwrite(app.config['STATIC_FOLDER'] + filename.replace('.dcm', '.png'), gray)
                image_src = os.path.join(app.config['STATIC_FOLDER'], filename.replace('.dcm', '.png'))
        else:
            image_src = ""
        session['update_cases'] = True
        ners=current_problem.get_problem().get_term_list()
        ners={n.replace('xxxx',''):v for n,v in ners.items()}
        return render_template('cbr_retrieve.html', content=current_problem.get_problem().get_report(),
                               image_src=image_src,
                               ners=ners)

    elif choice == "modify":
        return render_template('modify_solution.html', sectioned_report=sectioned_report_to_dict,
                               sugg_problems=current_problem.get_solution().get_sugg_ner()['problem'],
                               sugg_treatments=current_problem.get_solution().get_sugg_ner()['treatment'],
                               sugg_tests=current_problem.get_solution().get_sugg_ner()['test'],
                               sugg_abbrv=current_problem.get_solution().get_sugg_abbvs(),
                               related_cases=scored_cases)
    else:
        return view_case(choice)


@app.route("/modify_case", methods=['POST'])
def modify_case():
    global current_problem
    cases = request.form.getlist('selected_cases')
    report = request.form['modify_report']
    sectioned_report_to_dict = it.section_string_to_dict(report.rstrip().split('\n'))
    new_cases = [case for case in current_problem.get_solution().get_related_cases() if case in cases]
    current_problem.get_solution().set_related_cases(new_cases)
    global scored_cases
    rel_cases=[case for case in scored_cases if case['Case_ID'] in new_cases]
    st_name=storage_unit.get_case_host()
    if st_name != 'externals/ecgen-samples/case_set/' and st_name!='externals/mimic-cxr-samples/case_set/':
        storage_unit.dump_case(current_problem)
        storage_unit.link_similar_n_cases()
    # Clear current case so that is empty
    # Remove data in static
    # if current_problem.get_problem().get_image_file():
    #     filename = current_problem.get_problem().get_image_file()[0].split('/')[-1]
    #     shutil.copy(os.path.join(app.config['UPLOAD_FOLDER'], filename),
    #                 os.path.join(storage_unit.get_originals_host() + '/images/', filename))
    #     os.remove(os.path.join(app.config['STATIC_FOLDER'], filename))
    #     os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    flash("Success!")
    return render_template('modify_solution.html', sectioned_report=sectioned_report_to_dict,
                           sugg_problems=current_problem.get_solution().get_sugg_ner()['problem'],
                           sugg_treatments=current_problem.get_solution().get_sugg_ner()['treatment'],
                           sugg_tests=current_problem.get_solution().get_sugg_ner()['test'],
                           sugg_abbrv=current_problem.get_solution().get_sugg_abbvs(),
                           related_cases=rel_cases)


@app.route("/view_case", methods=['POST'])
def view_case(case_id=None):
    choice = request.form.get('view_case')
    if choice:
        case_id = choice
    case = storage_unit.recover_single_case(case_id)
    sectioned_report_to_dict = it.section_string_to_dict(case.get_solution().get_section_report().rstrip().split('\n'))
    static_image_files = []
    if case.get_problem().get_image_file():
        img_sources = case.get_problem().get_image_file()
        for img in img_sources:
            #View remote cases not valid
            filename = img.split('/')[-1]
            img_path = os.path.join(storage_unit.get_originals_host() + '/images/', filename)
            if server:
                storage_unit.download_image(filename)
                img_path='externals/tmp/'+filename
            if filename.endswith('.dcm') or filename.endswith('.DCM'):
                dicom_image = pydicom.dcmread(img_path)
                arr = dicom_image.pixel_array
                rgb_image = apply_color_lut(arr, dicom_image, palette='PET')
                gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
                cv2.imwrite(app.config['STATIC_FOLDER'] + filename.replace('.dcm', '.png'), gray)
                img_src = os.path.join(app.config['STATIC_FOLDER'], filename.replace('.dcm', '.png'))
            else:
                shutil.copy(img_path, os.path.join(app.config['STATIC_FOLDER'], filename))
                img_src=os.path.join(app.config['STATIC_FOLDER'], filename)
            static_image_files.append(img_src)
    ners = it.get_entities(case.get_solution().get_section_report(), case.get_problem().get_term_list())
    ners={n.replace('xxxx',''):v for n,v in ners.items()}
    abbvs = case.get_solution().get_sugg_abbvs()
    related_cases = case.get_solution().get_related_cases()
    return render_template('view_case.html', sectioned_report=sectioned_report_to_dict,
                           image_src=static_image_files, ners=ners, sugg_abbvs=abbvs, related_cases=related_cases)


@app.route('/explore_cases')
def explore_cases():
    return render_template('explore_cases.html')


@app.route('/expert_settings', methods=['POST', 'GET'])
def expert_settings():
    choice = request.form.get('expert_choice')
    if not choice:
        return render_template('expert_settings.html')
    elif choice == 'validate_cases':
        return render_template('validate_cases.html')
    elif choice == 'create_case':
        return render_template('create_single_case.html')
    elif choice == 'create_case_set':
        return render_template('create_case_set.html')
    elif choice == 'link_cases':
        try:
            storage_unit.link_similar_n_cases()
            flash('All cases were successfully linked', 'info')
            return render_template('expert_settings.html')
        except Exception as e:
            flash(e, 'error')
            return render_template('expert_settings.html')
    elif choice == 'train_section':
        try:
            if len(os.listdir(storage_unit.get_case_host())) < 500:
                flash('You need at least 500 cases to train a robust section report', 'error')
            else:
                it.train_section_model(storage_unit.get_case_host())
                flash('Your sectioning model was successfully trained', 'info')
            return render_template('expert_settings.html')
        except Exception as e:
            print(e)
            return render_template('expert_settings.html')
    elif choice == 'train_score':
        try:
            if len(os.listdir(storage_unit.get_case_host())) < 100:
                flash('You need at least 100 cases to train a robust scoring report', 'error')
            else:
                it.train_section_model(storage_unit.get_case_host())
                flash('Your sectioning model was successfully trained', 'info')
            return render_template('expert_settings.html')
        except Exception as e:
            print(e)
            return render_template('expert_settings.html')


@app.route('/validate_cases', methods=['POST'])
def validate_cases():
    case_ids = []
    df = pd.read_pickle(storage_unit.get_csv_path())
    for i, row in df.iterrows():
        if row['Validation_Status'] == 'Pending':
            case_ids.append(row['Case_ID'])
    for c in case_ids:
        val_key = c + '_validation_value'
        validation_value = request.form[val_key]
        if validation_value != 'Pending':
            rep_key = c + '_modify'
            new_report = request.form[rep_key]
            it.cbr_revise(storage_unit, c, new_report, validation_value)
    return redirect((url_for('expert_settings')))


@app.route('/create_single_case', methods=['POST'])
def create_case():
    new_case = it.Case()
    new_report = request.form['input_report']
    image = request.files["image_file"]
    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        new_image = [os.path.join(app.config['UPLOAD_FOLDER'], image.filename)]
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    else:
        new_image = []
    section_x = request.form['roi_selected_x']
    section_y = request.form['roi_selected_y']
    section_h = request.form['roi_selected_h']
    section_w = request.form['roi_selected_w']
    if section_x and section_y and section_h and section_w:
        rois = [int(section_x), int(section_y), int(section_w), int(section_h)]
    else:
        rois = []

    terms = request.form['ne_terms'].rstrip().split(',')
    abbvs = request.form['abbrvs'].rstrip().split(',')
    new_case.set_problem(new_report=new_report, new_image=new_image, new_rois=rois, new_terms=terms, new_abbrs=abbvs)
    section_report = request.form['section_report']
    storage_unit.create_new_case(new_case, section_report)
    flash('Your case was created successfully', 'info')
    return redirect((url_for('expert_settings')))


@app.route('/create_case_set', methods=['POST'])
def create_case_set():
    format = request.form['data_format']
    parsing_criteria = {'format': format, 'image_folder': storage_unit.get_originals_host() + '/images/'}
    if format == 'xml':
        report_tag = request.form['report_tag']
        parsing_criteria['report_headers'] = report_tag
        report_label = request.form.get('report_label')
        if report_label == 'yes':
            report_label = 'Label'
        else:
            report_label = request.form['report_label_no']
        parsing_criteria['report_label'] = report_label
        image_tag = request.form['image_tag']
        parsing_criteria['image_header'] = image_tag
        image_label = request.form.get('image_file')
        if image_tag:
            if image_label == 'yes':
                image_label = 'id'
            else:
                image_label = request.form['image_file_no']
        roi_tag=request.form['roi_tag']
        if roi_tag:
            parsing_criteria['roi_header']=roi_tag
            roi_coordinates=request.form.get('roi_coordinates')
            if roi_coordinates == 'yes':
                roi_coordinates='coordinate'
            else:
                roi_coordinates=request.form['roi_value_no']
        parsing_criteria['roi_coordinate']=roi_coordinates
        parsing_criteria['image_label'] = image_label
        parsing_criteria['NE_headers'] = request.form['term_tag']
        parsing_criteria['abbv_headers']=request.form['abbv_tag']
    storage_unit.create_case_set(parsing_criteria)
    flash('Your cases were successfully created!', 'info')
    return render_template(url_for('expert_settings'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

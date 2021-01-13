class _Solution:
    def __init__(self):
        self.sections = """"""
        self.sugg_abbvs = {}
        self.sugg_ner = {}
        self.similar_cases = []
        self.original_section=""""""

    def get_data(self):
        return self.sections, self.sugg_abbvs, self.sugg_ner, self.similar_cases

    def set_data(self, new_sections, new_abbvs, new_ner, new_cases):
        self.sections = new_sections
        self.sugg_abbvs = new_abbvs
        self.sugg_ner = new_ner
        self.similar_cases = new_cases

    def set_original_section(self,new_original_section):
        self.original_section=new_original_section

    def set_related_cases(self,new_cases):
        self.similar_cases=new_cases

    def set_suggested_abbs(self,new_abbvs):
        self.sugg_abbvs=new_abbvs

    def set_suggested_ner(self,new_ners):
        self.sugg_abbvs=new_ners

    def set_section_report(self,new_report):
        self.sections=new_report

    def get_section_report(self):
        return self.sections

    def get_sugg_abbvs(self):
        return self.sugg_abbvs

    def get_sugg_ner(self):
        return self.sugg_ner

    def get_related_cases(self):
        return self.similar_cases

    def get_original_section(self):
        return self.original_section


class _Problem:
    def __init__(self):
        self.report = """"""
        self.image_files = []
        self.roi_coordinates = []
        self.term_list = []
        self.abbrs_employed = {}
        self.src_lang = 'en'
        self.original_report = """"""

    def get_data(self):
        return self.report, self.image_files, self.roi_coordinates, self.term_list, self.abbrs_employed

    def get_report(self):
        return self.report

    def get_src_lang(self):
        return self.src_lang

    def get_original_report(self):
        return self.original_report

    def get_image_file(self):
        return self.image_files

    def get_roi_coordinates(self):
        return self.roi_coordinates

    def get_term_list(self):
        return self.term_list

    def get_abbrs(self):
        return self.abbrs_employed

    def set_data(self, new_report, new_image, new_rois, new_terms, new_abbrs):
        self.report = new_report
        self.image_files = new_image
        self.roi_coordinates = new_rois
        self.term_list = new_terms
        self.abbrs_employed = new_abbrs

    def set_lang_data(self,new_lang, new_original_report):
        self.src_lang=new_lang
        self.original_report=new_original_report

    def set_report(self,new_report):
        self.report=new_report

    def set_image_files(self,new_image_files):
        self.image_files=new_image_files

    def set_rois(self,new_rois):
        self.roi_coordinates=new_rois

    def set_term_list(self,new_terms):
        self.term_list=new_terms

    def set_abbrs(self,new_abbrvs):
        self.abbrs_employed=new_abbrvs


class Case:
    def __init__(self):
        self.problem = _Problem()
        self.solution = _Solution()

    def set_problem(self, new_report="""""", new_image=[], new_rois=[], new_terms=[], new_abbrs={}):
        self.problem.set_data(new_report, new_image, new_rois, new_terms, new_abbrs)

    def get_problem(self):
        return self.problem

    def set_solution(self, new_sections="""""", new_abbvs={}, new_ner={}, new_cases=[]):
        self.solution.set_data(new_sections, new_abbvs, new_ner, new_cases)

    def get_solution(self):
        return self.solution

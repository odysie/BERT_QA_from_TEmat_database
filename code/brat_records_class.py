from TE_databse_to_QA import pick_one_from_synonyms, get_unfindable_datapoint
import random
import re

class BRAT_record:
    """
    Class to represent a BRAT annotation record with fields for raw value, specifier, names, temperature, 
    and sentence. It supports methods for processing the record into QA format for the test dataset.
    """
        
    def __init__(self, raw_value, specifier, names, temperature, sentence, start_indices_for_record):
        """
        Initialize a BRAT_record instance.

        Args:
            raw_value (str): The raw value and units of the record.
            specifier (list): A list of specifiers for the record.
            names (list): A list of names or materials associated with the record.
            temperature (str): The temperature at which the measurement was taken.
            sentence (str): The full sentence context of the record.
            start_indices_for_record (dict): A dictionary mapping record elements to their start indices in the text.
        """
        self.pattern = r'[–−—―]'

        self.equals_check_fields = "names raw_value_and_units any_temperature".split()

        assert isinstance(specifier, list), "specifier should be a list now"
        
        self.raw_value_and_units = re.sub(self.pattern, '-', raw_value)
        self.specifier = [re.sub(self.pattern, '-', s) for s in specifier]
        self.names = [re.sub(self.pattern, '-', n) for n in names]  # main difference 1/2 to CDED_record
        self.any_temperature = re.sub(self.pattern, '-', temperature)
        self.sentence = re.sub(self.pattern, '-', sentence)

        self.start_indices_for_record = start_indices_for_record  # main difference 2/2, specific to BRAT records

    def __str__(self):

        return f"specifiers: {self.specifier},\nnames: {self.names},\nraw value: {self.raw_value_and_units},\
        \ntemperature: {self.any_temperature},\nsentence: {self.sentence[:22]}...\nstartindices_dictionary: {self.start_indices_for_record}"

    def get_startindices_from_answers(self):

        temperature_answer_start = self.start_indices_for_record[self.any_temperature]
        specifier_answer_starts = [self.start_indices_for_record[specifier] for specifier in self.specifier]
        material_answer_starts = [self.start_indices_for_record[name] for name in self.names]
        return [[temperature_answer_start], specifier_answer_starts, material_answer_starts]

    def process_valunits(self, raw_value_and_units):

        return f"{raw_value_and_units.strip()}".replace("∼ ", "∼")

    def process_temperature(self, any_temperature):

        return f"{any_temperature.strip()}".replace("∼ ", "∼")

    def to_QA_for_test_dataset(self):
        """
        Generate question-answer pairs for the test dataset.

        Returns:
            tuple: A tuple containing lists of questions, answers, and answer start indices.
        """
        A1_valunits = self.process_valunits(self.raw_value_and_units)
        Q2_temperature = f"At what temperature was the value of {A1_valunits} recorded?"
        A2_temperature = self.process_temperature(self.any_temperature)
        Q3_specifier = f"Which property was recorded to be {A1_valunits} at {A2_temperature}?"
        A3_specifier_synonyms = self.specifier  # list of synonyms
        A3_selected_specifier = pick_one_from_synonyms(A3_specifier_synonyms)
        Q4_material = f"Which material was recorded to have a {A3_selected_specifier} of {A1_valunits} at {A2_temperature}?"
        A4_material_synonyms = self.names  # list of synonyms

        questions = [Q2_temperature, Q3_specifier, Q4_material]
        answers = [[A2_temperature], A3_specifier_synonyms, A4_material_synonyms]
        answer_starts = self.get_startindices_from_answers()

        return questions, answers, answer_starts

    def to_unanswerable_QA_for_test_dataset(self, context):
        """
        Generate unanswerable question-answer pairs for the test dataset.

        Args:
            context (str): The context in which to generate unanswerable questions.

        Returns:
            tuple: A tuple containing lists of unanswerable questions, empty answers, and -1 answer start indices.
        """
        unanswerable_questions = []

        A1_valunits = self.process_valunits(self.raw_value_and_units)
        A2_temperature = self.process_temperature(self.any_temperature)
        A3_specifier_synonyms = self.specifier
        A3_selected_specifier = pick_one_from_synonyms(A3_specifier_synonyms)

        # Q2: unfindable value and units -> looking for temperature
        unfindable_valunits = get_unfindable_datapoint(A1_valunits, "value_and_units", context)
        unanswerable_Q2_temperature = f"At what temperature was the value of {unfindable_valunits} recorded?"
        unanswerable_questions.append(unanswerable_Q2_temperature)

        # Q3: unfindable value and units, or temperature -> looking for specifier
        sabotage_choice = random.choice([1,2])
        if sabotage_choice == 1:
            unfindable_valunits = get_unfindable_datapoint(A1_valunits, "value_and_units", context)
            unanswerable_Q3_specifier = f"Which property was recorded to be {unfindable_valunits} at {A2_temperature}?"
        else:
            unfindable_temperature = get_unfindable_datapoint(A2_temperature, "temperature", context)
            unanswerable_Q3_specifier = f"Which property was recorded to be {A1_valunits} at {unfindable_temperature}?"
        unanswerable_questions.append(unanswerable_Q3_specifier)

        # Q4: unfindable value and units, temperature, or specifier -> looking for material
        sabotage_choice = random.choice([1,2,3])
        if sabotage_choice == 1:
            unfindable_valunits = get_unfindable_datapoint(A1_valunits, "value_and_units", context)
            unanswerable_Q4_material = f"Which material was recorded to have a {A3_selected_specifier} of {unfindable_valunits} at {A2_temperature}?"
        elif sabotage_choice == 2:
            unfindable_temperature = get_unfindable_datapoint(A2_temperature, "temperature", context)
            unanswerable_Q4_material = f"Which material was recorded to have a {A3_selected_specifier} of {A1_valunits} at {unfindable_temperature}?"
        else:
            unfindable_specifier = get_unfindable_datapoint(A3_selected_specifier, "specifier", context)
            unanswerable_Q4_material = f"Which material was recorded to have a {unfindable_specifier} of {A1_valunits} at {A2_temperature}?"
        unanswerable_questions.append(unanswerable_Q4_material)

        no_answers = ["" for _ in unanswerable_questions]
        no_answerstarts = [-1 for _ in unanswerable_questions]
        return unanswerable_questions, no_answers, no_answerstarts

    def __eq__(self, other):

        return all([getattr(self, field) == getattr(other, field) for field in self.equals_check_fields])

from analysis.data_analyser import DataAnalyser, OutputAnalysis
from utils.predictor import DPMPredictor
from utils.data_loader import PCLDatasetLoader
from sklearn.metrics import classification_report
from utils import config
import pandas as pd
from utils.data_loader import FINAL_LABEL_MAP

def load_output():
    output_dev = PCLDatasetLoader('../data/output.csv')._load_dataset()
    dpm = PCLDatasetLoader(config.DATA_PATH)._load_dataset()
    dpm['ground_labels'] = dpm['labels']
    dpm = dpm.drop('labels', axis=1)
    data = pd.merge(dpm, output_dev, on='texts', how='inner')
    data = data.dropna() 
    data['preds'] = data['preds'].apply(lambda p: int(p[1]))
    return data

if __name__ == '__main__':
    ### Uncomment to produce the plots for the report
    # dpm_analyser = DataAnalyser('dpm')
    # dpm_analyser.plot_word_cloud() # Plot Word Cloud
    # dpm_analyser.get_average_lengths()
    # print(dpm_analyser.get_capital_count())
    # print(dpm_analyser.get_sentences_count())
    # dpm_analyser.get_analysis_of_class_labels()

    # dpm_analyser_ = DataAnalyser('dpm', map_to_binary=True)
    # dpm_analyser_.get_distribution_analysis()
    # dpm_analyser_.chi_test()
    # dpm_analyser_.get_analysis_by_country_n_topic()

    ### Beware of the warnings due to division by zero!
    df = load_output()
    f1_by_gl = OutputAnalysis.compute_f1_by_ground_label(df)
    f1_by_text_length = OutputAnalysis.compute_f1_by_text_length(df)
    f1_by_topic = OutputAnalysis.compute_f1_by_topic(df)
    f1_by_country = OutputAnalysis.compute_f1_by_country(df)

    OutputAnalysis.plot_f1_by_category_grouped(f1_by_text_length, 'Text Length', 'F1 Score', 'F1 Score by Text Length', save_fig_name='f1_text')
    OutputAnalysis.plot_f1_by_category(f1_by_topic, 'Topic', 'F1 Score', 'F1 Score by Topic', save_fig_name='f1_topic', rotation=30)
    OutputAnalysis.plot_f1_by_category(f1_by_country, 'Country', 'F1 Score', 'F1 Score by Country', save_fig_name='f1_country')
    OutputAnalysis.plot_f1_by_category(f1_by_gl, 'PCL Severity', 'F1 Score', 'F1 Score by PCL Severity', rotation=None, save_fig_name='f1_pcl')
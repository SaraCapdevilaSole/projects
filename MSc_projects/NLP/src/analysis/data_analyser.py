import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import config
from utils.data_loader import PCLDatasetLoader, PCLDataHandler, FINAL_LABEL_MAP
from analysis.utils import AnalysisUtils
from pprint import pprint
from scipy.stats import chi2_contingency
from sklearn.metrics import f1_score

class DataAnalyser(AnalysisUtils):
    def __init__(self, data_type: str = 'dpm', map_to_binary=False) -> None:
        assert data_type in ['dpm'], f'unidentified data_type: {data_type}'
        if data_type == 'dpm':
            self.data = PCLDatasetLoader(file_path=config.DATA_PATH)._load_tsv()
        # if data_type == 'train_eval_test':
            # train, dev, test
            # self.data = PCLDataHandler().get_datasets()
        if map_to_binary:
            self.data['labels'] = self.data['labels'].map(FINAL_LABEL_MAP)
            self.columns = list(set(FINAL_LABEL_MAP.values()))
        else:
            self.columns = FINAL_LABEL_MAP.keys()
        super().__init__(self.data)


    def chi_test(self):
        df = self.data
        
        contingency_country = pd.crosstab(df['labels'], df['country'])
        ratio = contingency_country.sum(axis=1) 
        contingency_country.loc[1, :] *= float(ratio[0]/ratio[1])

        #contingency_country = contingency_country.div(contingency_country.sum(axis=1), axis=0)
        res_country = chi2_contingency(contingency_country)

        contingency_topic = pd.crosstab(df['labels'], df['topic'])
        ratio = contingency_topic.sum(axis=1) 
        contingency_topic.iloc[1,:] *= float(ratio[0]/ratio[1])
        # contingency_topic = contingency_topic.div(contingency_topic.sum(axis=1), axis=0)
        res_topic = chi2_contingency(contingency_topic)

        # Display the results
        print(f"Chi-square test for 'labels' and 'country':")
        print(f"Chi2 value: {res_country.statistic}, p-value: {res_country.pvalue}")
        print("\n")
        print(f"Chi-square test for 'labels' and 'topic':")
        print(f"Chi2 value: {res_topic.statistic}, p-value: {res_topic.pvalue}")

    def plot_word_cloud(self):
        for c in self.columns:
            self._generate_word_cloud(c) 

    def get_average_lengths(self):
        return {
            c: self._average_length(c) 
            for c in self.columns
            }
    
    def get_capital_count(self):
        return {
            c: self._capital_count(c) 
            for c in self.columns
            }
    
    def get_sentences_count(self):
        return {
            c: self._sentences_count(c) 
            for c in self.columns
            }
    
    def get_distribution(self):
        return {
            c: self._get_distribution(c)
            for c in self.columns
        }
    
    def get_analysis_of_class_labels(self):
        """Analysis of the class labels: how frequent these are and how they correlate with any feature of the data, e.g. input length."""
        
        print('Average length per class label:')
        avg_lengths = self.get_average_lengths()
        pprint(avg_lengths)
        self.plot_bar(avg_lengths)

        print('Number of capitals used per class label:')
        capital_count = self.get_capital_count()
        pprint(capital_count)
        self.plot_bar_caps(capital_count)

        print('Number of sentences per class label:')
        sentences_count = self.get_sentences_count()
        pprint(sentences_count)
        self.plot_bar(sentences_count, ylabel='Number of Sentences')

    def get_ratio_per_country(self):
        return {
            c: self._ratio_per_filter(c, filter_by='country') 
            for c in self.columns
            }
    
    def get_ratio_per_topic(self):
        return {
            c: self._ratio_per_filter(c, filter_by='topic') 
            for c in self.columns
            }
    
    def get_sentence_length_per_country(self):
        return {
            c: self._sentence_length_by_filter(c, filter_by='country') 
            for c in self.columns
            }
    
    def get_sentence_length_per_topic(self):
        return {
            c: self._sentence_length_by_filter(c, filter_by='topic') 
            for c in self.columns
            }
    
    def get_sentence_count_per_country(self):
        return {
            c: self._sentence_count_by_filter(c, filter_by='country') 
            for c in self.columns
            }
    
    def get_sentence_count_per_topic(self):
        return {
            c: self._sentence_count_by_filter(c, filter_by='topic') 
            for c in self.columns
            }
    
    def get_analysis_by_country_n_topic(self):
        """Analysis of country and topic by label"""
        print("Ratio for each country, by label:")
        ratios = self.get_ratio_per_country()
        pprint(ratios)
        self.h_bar_plot(ratios, ylabel='Country', xlabel='%', save_fig_name='ratio_by_country.png')
        
        print("Ratio of topics by label")
        topics_ratios = self.get_ratio_per_topic()
        pprint(topics_ratios)
        self.h_bar_plot(topics_ratios, ylabel='Topics', xlabel='%', save_fig_name='ratio_by_topic.png')

        print("Average sentence length by topic")
        length_topic = self.get_sentence_length_per_topic()
        pprint(length_topic)
        self.h_bar_plot(length_topic, ylabel='Topics', xlabel='Average sentence length', save_fig_name='avg_length_by_topic.png')

        print("Number of sentences by topic")
        count_topic = self.get_sentence_count_per_topic()
        pprint(count_topic)
        self.h_bar_plot(count_topic, ylabel='Topics', xlabel='Sentence count', save_fig_name='sentence_count_by_topic.png')

        print("Average sentence length by country")
        length_country = self.get_sentence_length_per_country()
        pprint(length_country)
        self.h_bar_plot(length_country, ylabel='Country', xlabel='Average sentence length', save_fig_name='avg_length_by_country.png')

        print("Number of sentences by country")
        count_country = self.get_sentence_count_per_country()
        pprint(count_country)
        self.h_bar_plot(count_country, ylabel='Country', xlabel='Sentence count', save_fig_name='sentence_count_by_country.png')

    def get_distribution_analysis(self):
        print('Number of occurances:')
        distribution = self.get_distribution()
        pprint(distribution)
        self.plot_bar(distribution, ylabel='Count')

class OutputAnalysis:
    def compute_f1_by_category(df, category_column, category_values, pcl_only=False):
        f1_by_category = {}

        for value in category_values:
            if pcl_only:
                if value in [0,1]:
                    continue
            value_df = df[df[category_column] == value]
            f1 = f1_score(value_df['labels'], value_df['preds'], pos_label=1)
            f1_by_category[value] = f1
        return dict(sorted(f1_by_category.items())) 

    def compute_f1_by_text_length(df):
        df['text_length'] = df['texts'].apply(lambda line: len(line.split(" ")))
        return OutputAnalysis.compute_f1_by_category(df, 'text_length', range(df['text_length'].min(), df['text_length'].max() + 1))

    def compute_f1_by_topic(df):
        return OutputAnalysis.compute_f1_by_category(df, 'topic', df['topic'].unique())

    def compute_f1_by_country(df):
        return OutputAnalysis.compute_f1_by_category(df, 'country', df['country'].unique())

    def compute_f1_by_ground_label(df):
        return OutputAnalysis.compute_f1_by_category(df, 'ground_labels', df['ground_labels'].unique(), pcl_only=True)
        
    def plot_f1_by_category(f1_by_category, xlabel, ylabel, title, rotation=45, save_fig_name=None):
        if xlabel == 'Topic':
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_facecolor('#EFF0F1')  
        ax.grid(axis='y', linestyle='-', alpha=0.3, color='white')  
        
        plt.bar(f1_by_category.keys(), f1_by_category.values(), edgecolor='white', alpha=0.6)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        ax.tick_params(axis='both', labelsize=12) 
        plt.xticks(list(f1_by_category.keys()), rotation=rotation, ha='right')  
        plt.tick_params(axis='both', labelsize=12)
        plt.ylim([0,1])
        if save_fig_name:
            fig.savefig(f'../figs/{save_fig_name}.png', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.show()

    def plot_f1_by_category_grouped(f1_by_category, xlabel, ylabel, title, save_fig_name=None):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_facecolor('#EFF0F1')  

        ax.grid(axis='y', linestyle='-', alpha=0.3, color='white')  
        categories = list(f1_by_category.keys())
        f1_scores = list(f1_by_category.values())

        group_size = 10
        num_groups = len(categories) // group_size
        grouped_categories = [f'{categories[i * num_groups]} to {categories[(i + 1) * num_groups]}' for i in range(group_size)]
        grouped_f1_scores = [np.mean(f1_scores[i * num_groups:(i + 1) * num_groups]) for i in range(group_size)]

        plt.bar(grouped_categories, grouped_f1_scores, edgecolor='white', alpha=0.6)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(rotation=45, ha='right')  
        plt.tick_params(axis='both', labelsize=10)
        plt.ylim([0,1])

        plt.tight_layout()
        if save_fig_name:
            fig.savefig(f'../figs/{save_fig_name}.png', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.show()
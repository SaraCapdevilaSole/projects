from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import Union, List, Optional
import numpy as np

class PlottingUtils:
    @staticmethod
    def plot_img(img, title):
        plt.figure(figsize=(10, 6))
        plt.imshow(img, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.show()
    
    def plot_histogram():
        pass
    
    @staticmethod
    def plot_bar(dict, ylabel='Average Length'):
        plt.figure(figsize=(10, 6))
        plt.bar(dict.keys(), dict.values(), color='skyblue')
        plt.xticks(list(dict.keys()))
        plt.xlabel('Class Label')
        plt.ylabel(ylabel)
        plt.show()

    @staticmethod
    def plot_bar_caps(dict):
        plt.figure(figsize=(10, 6))
        caps_data = [dict[label]['caps'] for label in dict]
        noncaps_data = [dict[label]['noncaps'] for label in dict]
        # total_data = [dict[label]['total'] for label in dict]

        plt.bar(dict.keys(), caps_data, color='blue', label='Capitals')
        plt.bar(dict.keys(), noncaps_data, color='orange', label='Non-Capitals', bottom=caps_data)
        plt.xlabel('Class Label')
        plt.ylabel('Number of Capitals')
        plt.legend()
        plt.show()
    
    @staticmethod
    def h_bar_plot(dict, ylabel, xlabel, save_fig_name=None):
        fig, ax = plt.subplots(figsize=(10, 8))
        heights = {0: 0.6, 1:0.4}

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        ax.set_facecolor('#EFF0F1')  

        ax.grid(axis='x', linestyle='-', alpha=0.3, color='white')  

        for label, ratios in dict.items():
            ax.barh(ratios.index, ratios, label=f'label {label}', alpha=0.6, height=heights[label], edgecolor='white')

        if xlabel == '%':
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
            xlabel += ' of label'
        ax.set_xlabel(f'{xlabel}', fontsize=14) 
        ax.set_ylabel(ylabel,  fontsize=14) 
        ax.legend(fontsize=14) 
        ax.tick_params(axis='both', labelsize=12) 

        if save_fig_name:
            fig.savefig(f'../figs/{save_fig_name}.png', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.show()


class AnalysisUtils(PlottingUtils):  
    def __init__(self, data) -> None:
        # super().__init__()
        data = data.dropna() # clean
        self.data = data
        self.word_corpus = self._df_to_list(data, None)

    @staticmethod
    def _df_to_list(df, label: Optional[Union[int, List]] = None):
        if not label:
            data = df['texts'].tolist()
        elif isinstance(label, int): 
            data = df[df['labels'] == label]['texts'].tolist()
        elif isinstance(label, list):  
            data = df[df['labels'].isin(label)]['texts'].tolist()
        return data

    def _generate_corpus(self, df, label: Union[int, List, bool]):
        data = self._df_to_list(df, label)
        text_for_label = ' '.join(data)
        return text_for_label
    
    def _generate_word_cloud(self, label):
        text_for_label = self._generate_corpus(self.data, label)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_for_label)
        self.plot_img(img=wordcloud, title=f'Word Cloud for Label {label}')

    def _average_length(self, label):
        data_list = self._df_to_list(self.data, label)
        text_length = np.mean([self._get_length(line) for line in data_list])
        return text_length
        
    @staticmethod
    def _get_length(line):
        text = line.split(" ")
        return len(text)
    
    @staticmethod
    def _get_capital_count(line):
        text = line.split(" ")
        capital_count = 0
        noncapital_count = 0
        for char in text:
            if char[0].isupper():
                capital_count += 1
            else:
                noncapital_count +=1
        return np.array([capital_count, noncapital_count])

    def _capital_count(self, label):
        count = np.zeros(2)
        data_list = self._df_to_list(self.data, label)
        for line in data_list:
            count += self._get_capital_count(line)
        return {
            'caps': count[0], 
            'noncaps': count[1],
            'total': np.sum(count),
            'proportion': count[0]/np.sum(count)*100
        }
    
    def _sentences_count(self, label):
        data_list = self._df_to_list(self.data, label)
        return len(data_list)

    def _ratio_per_filter(self, label, filter_by='country'):
        total = self.data[filter_by].value_counts()
        samples = self.data[self.data['labels'] == label][filter_by].value_counts()
        return samples / total
    
    def _sentence_length_by_filter(self, label, filter_by='country'):
        data = self.data.copy()
        data = data[data['labels']==label]
        data['sentence_length'] = data['texts'].apply(lambda x: self._get_length(x))
        return data.groupby(filter_by)['sentence_length'].mean()
        
    def _sentence_count_by_filter(self, label, filter_by='country'):
        data = self.data.copy()
        data = data[data['labels'] == label]
        return data.groupby(filter_by)['texts'].count()

    def _get_distribution(self, label):
        return len(self.data[self.data['labels']==label])



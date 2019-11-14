from pyspark.ml import Transformer
from pyspark.sql.functions import udf
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType, ArrayType
import re

class StopWordsCustom(Transformer):
    """
    A custom Transformer that...transform words!!!! :-)
    """

    def __init__(self, inputCol = None, outputCol = None):
        super(StopWordsCustom, self).__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol

    def setParams(self, inputCol = None, outputCol = None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)


    def _transform(self, df: DataFrame) -> DataFrame:

        def get_neg_words_list():
            neg_words_list = [  "aren't", 'couldn', "couldn't", 'didn', "don't", "can't"
                                "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
                                "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustnt'",
                                'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasnt'",
                                'weren', "werent'", 'won', "won't", 'wouldn', "wouldn't"]

            return neg_words_list
        
        def get_stop_words_list():
            stop_words_list = ['link','google','facebook','yahoo','rt','i', 'me', 'my', 'myself', 'tag'
                              'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                              "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                              'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its',
                              'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                              'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
                              'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
                              'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
                              'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
                              'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                              'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                              'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                              'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                              'only', 'own', 'same', 'so', 'than', 'too', 's', 't', 'can', 'will',
                              'just', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',
                              'y', 'ain', 'ma', 'u', 'aren', 'ø', 'å', 'æ', 'b', 'c', 'd', 'e']

            return stop_words_list


        def stop_words(review):
            text_line = []
            words = review.split()
            for word in words:
                if word in get_neg_words_list():
                    text_line.append('not')
                elif word not in get_stop_words_list():
                    text_line.append(word)
            return ' '.join(text_line)

        def pos_processing(review):            
            review = stop_words(review)
            review = re.sub(r"[.,:;']", '', review)
            return list(review.split())

        udf_c = udf(pos_processing, ArrayType(StringType()))
        df = df.withColumn(self.outputCol, udf_c(self.inputCol))
        return df
from pyspark.ml import Transformer
from pyspark.sql.functions import udf
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType, ArrayType
from pyspark.sql.functions import *
import numpy as np
import re

class PosProcessingGeneral(Transformer):
    """
    A custom Transformer that...transform words!!!! :-)
    """

    def __init__(self, inputCol = None, outputCol = None):
        super(PosProcessingGeneral, self).__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol

    def setParams(self, inputCol = None, outputCol = None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)


    def _transform(self, df: DataFrame) -> DataFrame:

        def get_neg_words_list():
            neg_words_list = [  "arent", 'couldn', "couldnt", 'didn', "dont", "cant"
                                "didnt", 'doesn', "doesnt", 'hadn', "hadnt", 'hasn', "hasnt", 'haven',
                                "havent", 'isn', "isnt", 'ma', 'mightn', "mightnt", 'mustn', "mustnt",
                                'needn', "neednt", 'shan', "shant", 'shouldn', "shouldnt", 'wasn', "wasnt",
                                'weren', "werent", 'won', "wont", 'wouldn', "wouldnt"]

            return neg_words_list


        def stop_words(review):
            text_line = []
            words = review.split()
            for word in words:
                if word in get_neg_words_list():
                    text_line.append('not')
                else:
                    text_line.append(word)

            return ' '.join(text_line)

        def pos_processing(review):
            review = re.sub(r"[.,:;']", '', review)
            review = stop_words(review)
            return list(review.split())

        udf_c = udf(pos_processing, ArrayType(StringType()))
        df = df.withColumn(self.outputCol, udf_c(self.inputCol))
        return df
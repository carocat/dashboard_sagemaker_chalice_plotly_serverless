from pyspark.ml import Transformer
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf
import itertools
from itertools import groupby
import re

class PreProcessingGeneral(Transformer):


    """
    A custom Transformer that...transform words!!!! :-)
    """

    def __init__(self, inputCol = None, outputCol = None):
        super(PreProcessingGeneral, self).__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol

    def setParams(self, inputCol = None, outputCol = None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)


    def _transform(self, df: DataFrame) -> DataFrame:

        def pre_processing_general(review):   

            # mark emoticons as happy or sad
            review = re.sub(r'([xX;:]-?[dD)]|:-?[\)]|[;:][pP])', ' happyemoticons ', review)
            review = re.sub(r"(:'?[/|\(])", ' sademoticons ', review)
            
            #replace repetition chars and keep only one 'halo there this is an example'
            review = ''.join(c[0] for c in itertools.groupby(review.lower()))

            #replace duplicate words
            no_dupes = ([k for k, v in groupby(review.split())])
            print('No duplicates:', no_dupes)
            review = ' '.join(no_dupes)

            #remove numbers, fractions, etc...
            review =  re.sub(r"[-]?[0-9]+[,.]?[0-9]*([\/][0-9]+[,.]?[0-9]*)*", '', review)

            

            # delete mentions the mentions
            review = re.sub(r'@[a-zA-Z0-9_]* ', "", review)

            # Keeping only the word after the #
            review = re.sub(r'#', "", review)
            review = re.sub(r'[-\n]', '', review)

            # Removing links
            review = re.sub(r"https?:\S*", '', review)
            review = re.sub(r'http?:\S*', '', review)

            #remove punctuaction except delimiters ?.,!:;
            review =  re.sub(r"[$%&()*+/<=>@[\]^_`{|}~]", '', review)


            return " ".join(review.split())

        udf_c = udf(pre_processing_general)
        df = df.withColumn(self.outputCol, udf_c(self.inputCol))
        return df
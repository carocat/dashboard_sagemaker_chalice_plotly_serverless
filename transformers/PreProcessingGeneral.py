from pyspark.ml import Transformer
from pyspark.sql.functions import udf
from pyspark.sql import DataFrame
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

            HAPPY_EMO = r'([xX;:]-?[dD)]|:-?[\)]|[;:][pP])'
            SAD_EMO = r" (:'?[/|\(]) "

            # mark emoticons as happy or sad
            review = re.sub(HAPPY_EMO, ' happyemoticons ', review)
            review = re.sub(SAD_EMO, ' sademoticons ', review)

            # delete mentions the mentions
            review = re.sub(r'@[a-zA-Z0-9_]* ', "", review)

            # Keeping only the word after the #
            review = re.sub(r'#', "", review)
            review = re.sub(r'[-\n]', '', review)

            # Removing links
            review = re.sub(r"https?:\S*", "", review)
            review = re.sub(r'http?:\S*', '', review)

            #remove punctuaction except delimiters ?.,!:;
            review =  re.sub(r"[$%&()*+/<=>@[\]^_`{|}~]", '', review)

            return " ".join(review.split())

        udf_c = udf(pre_processing_general)
        df = df.withColumn(self.outputCol, udf_c(self.inputCol))
        return df
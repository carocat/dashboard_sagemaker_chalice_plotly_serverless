
from pyspark.ml import Transformer
from string import digits
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from pyspark.sql.functions import udf
from pyspark.sql import DataFrame

class LemmatizeStemWords(Transformer):
    """
    A custom Transformer that...transform words!!!! :-)
    """

    def __init__(self, inputCol = None, outputCol = None):
        super(LemmatizeStemWords, self).__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol

    def setParams(self, inputCol = None, outputCol = None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)


    def _transform(self, df: DataFrame) -> DataFrame:

        #lemmatize, steam and remove custom stop words in text
        def lemmatize_stem_stop_words(review):
            lemmatizing = WordNetLemmatizer()
            stemming = PorterStemmer()
            lower_case_split = review.split()
            text_line = []
            for word in lower_case_split:
                #lemmatizing
                    lemmatized = lemmatizing.lemmatize(word)
                    #steamming
                    steammed = stemming.stem(lemmatized)
                    text_line.append(steammed)            
            return ' '.join(text_line)
    


        udf_c = udf(lemmatize_stem_stop_words)
        df = df.withColumn(self.outputCol, udf_c(self.inputCol))
        return df
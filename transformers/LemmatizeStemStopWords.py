
from pyspark.ml import Transformer
from string import digits
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from pyspark.sql.functions import udf
from pyspark.sql import DataFrame

class LemmatizeStemStopWords(Transformer):
    """
    A custom Transformer that...transform words!!!! :-)
    """

    def __init__(self, inputCol = None, outputCol = None):
        super(LemmatizeStemStopWords, self).__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol

    def setParams(self, inputCol = None, outputCol = None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)


    def _transform(self, df: DataFrame) -> DataFrame:

        #lemmatize, steam and remove custom stop words in text
        def lemmatize_stem_stop_words(review):
            lower_case = review.lower()
            remove_digits = str.maketrans('', '', digits)
            lemmatizing = WordNetLemmatizer()
            stemming = PorterStemmer()
            lower_case_split = lower_case.split()
            text_line = []
            for review in lower_case_split:
                #remove digits
                res = review.translate(remove_digits)
                words = res.split()
                for word in words:
                    #lemmatizing
                    lemmatized = lemmatizing.lemmatize(word)
                    #steamming
                    steammed = stemming.stem(lemmatized)
                    text_line.append(steammed)
            return ' '.join(text_line)


        udf_c = udf(lemmatize_stem_stop_words)
        df = df.withColumn(self.outputCol, udf_c(self.inputCol))
        return df
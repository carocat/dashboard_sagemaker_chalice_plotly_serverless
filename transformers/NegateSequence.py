from pyspark.ml import Transformer
from string import digits
from pyspark.sql.functions import udf
from pyspark.sql import DataFrame

class NegateSequence(Transformer):
    """
    A custom Transformer that...transform words!!!! :-)
    """

    def __init__(self, inputCol = None, outputCol = None):
        super(NegateSequence, self).__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol

    def setParams(self, inputCol = None, outputCol = None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)


    def _transform(self, df: DataFrame) -> DataFrame:

        #highlight negative sentences
        def negate_sequence(review):
            negation = False
            delims = "?.,!:;"
            text_line = []
            remove_digits = str.maketrans('', '', digits)
            words = review.split()
            for word in words:
                res = word.translate(remove_digits)
                stripped = res.strip(delims).lower()
                negated = "nott_" + res if negation else stripped
                text_line.append(negated)
                if any(neg in word for neg in ["n't"]) or word in ['no', 'not']:
                    negation = True
                if any(c in word for c in delims):
                    negation = False
            return ' '.join(text_line)

        udf_c = udf(negate_sequence)
        df = df.withColumn(self.outputCol, udf_c(self.inputCol))
        return df

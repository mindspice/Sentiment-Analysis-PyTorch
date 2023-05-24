import re
import contractions


#spacy.cli.download("en_core_web_lg")



class PreProcessor:

    def preprocess_text(self, text):
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove @handle mentions
        text = re.sub(r'@\w+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Expand contractions
        text = contractions.fix(text)
        # Remove non-alphanumeric characters and convert to lowercase
        text = re.sub(r'\W', ' ', text).lower()
        return text







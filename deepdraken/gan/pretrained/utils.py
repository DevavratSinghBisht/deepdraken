import json
import numpy as np
from collections import OrderedDict
from sentence_transformers import SentenceTransformer

class EmbeddingMaker:

    def __init__(self, model = 'bert-base-nli-mean-tokens'):
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')

    def create_embeddings(self, label_file, embedding_file, index_to_class_file, class_to_index_file):

        with open(label_file) as f:
            labels = json.load(f)

        embedding_dict = OrderedDict()
        embeddings = []
        for i in range(1000):
            sentences = labels[str(i)]
            sentence_embeddings = self.model.encode(sentences)
            embedding_dict[str(i)] = sentence_embeddings
            embeddings.extend(sentence_embeddings)

        embeddings = np.asarray(embeddings)

        class_to_index = OrderedDict()
        start = 0
        for key in embedding_dict:
            end = start + len(embedding_dict[key])
            class_to_index[key] = [str(i) for i in range(start, end)]
            start = end

        index_to_class = OrderedDict()    
        start = 0
        for key in embedding_dict:
            end = start + len(embedding_dict[key])
            for i in range(start, end):
                index_to_class[i] = key
            start = end

        with open(index_to_class_file, "w") as f:
            json.dump(index_to_class, f, indent=4)

        with open(class_to_index_file, "w") as f:
            json.dump(class_to_index, f, indent=4)

        np.save(embedding_file, embeddings)

class EmbeddingComparator(EmbeddingMaker):

    def __init__(self, embedding_file, index_to_class_file, model='bert-base-nli-mean-tokens'):
        
        super().__init__(model)
        
        self.embeddings = np.load(embedding_file)

        with open(index_to_class_file) as f:
            self.index_to_class = json.load(f)

    def cos_sim(a, b) -> np.ndarray:
        return np.dot(a, b)/(np.linalg.norm(a)* np.linalg.norm(b))

    def get_class_id(self, text, similarity="cosine") -> int:

        text_embedding = self.model.encode([text])
        
        if similarity == "cosine":
            return self.index_to_class[np.argmax(self.cos_sim(self.embeddings, text_embedding.T))]
        elif similarity == "dot":
            return self.index_to_class[np.argmax(np.dot(self.embeddings, text_embedding.T))]
        else:
            print(f"Unknown/unsupported similarity provided: {similarity}\n",
                  f"Please select from the options:\n",
                  f"    cosine: cosine similarity\n",
                  f"    dot   : dot similarity")

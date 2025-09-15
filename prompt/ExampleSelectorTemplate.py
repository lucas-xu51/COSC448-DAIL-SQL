import numpy as np
import random

from utils.utils import sql2skeleton, jaccard_similarity
from utils.linking_utils.application import mask_question_with_schema_linking


class BasicExampleSelector(object):
    def __init__(self, data, *args, **kwargs):
        self.data = data
        self.train_json = self.data.get_train_json()
        self.db_ids = [d["db_id"] for d in self.train_json]
        self.train_questions = self.data.get_train_questions()


    def get_examples(self, question, num_example, cross_domain=False):
        pass

    def domain_mask(self, candidates: list, db_id):
        cross_domain_candidates = [candidates[i] for i in range(len(self.db_ids)) if self.db_ids[i] != db_id]
        return cross_domain_candidates

    def retrieve_index(self, indexes: list, db_id):
        cross_domain_indexes = [i for i in range(len(self.db_ids)) if self.db_ids[i] != db_id]
        retrieved_indexes = [cross_domain_indexes[i] for i in indexes]
        return retrieved_indexes


class RandomExampleSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)
        random.seed(0)

    def get_examples(self, target, num_example, cross_domain=False):
        train_json = self.train_json
        indexes = list(range(len(train_json)))
        if cross_domain:
            indexes = domain_mask(indexes, target["db_id"])
        selected_indexes = random.sample(indexes, num_example)
        if cross_domain:
            selected_indexes = retrieve_index(selected_indexes, target["db_id"])
        return [train_json[index] for index in selected_indexes]


class CosineSimilarExampleSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = "sentence-transformers/all-mpnet-base-v2"
        # self.SELECT_MODEL = "sentence-transformers/bert-base-nli-mean-tokens"

        from sentence_transformers import SentenceTransformer
        self.bert_model = SentenceTransformer(self.SELECT_MODEL, device="cpu")
        self.train_embeddings = self.bert_model.encode(self.train_questions)

        
    def get_examples(self, target, num_example, cross_domain=False):
        target_embedding = self.bert_model.encode([target["question"]])
        # target_embedding = self.bert_model.embed_text([target["question"]]).cpu().detach().numpy()

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = np.squeeze(cosine_similarity(target_embedding, self.train_embeddings)).tolist()
        pairs = [(similarity, index) for similarity, index in zip(similarities, range(len(similarities)))]

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        top_pairs = list()
        for s, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            if train_json[index]["question"] == target["question"]:
                continue
            top_pairs.append((index, s))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, s) in top_pairs]


class EuclideanDistanceExampleSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = "sentence-transformers/all-mpnet-base-v2"

        from sentence_transformers import SentenceTransformer
        self.bert_model = SentenceTransformer(self.SELECT_MODEL, device="cpu")
        self.train_embeddings = self.bert_model.encode(self.train_questions)

    def get_examples(self, target, num_example, cross_domain=False):
        target_embedding = self.bert_model.encode([target["question"]])

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import euclidean_distances
        distances = np.squeeze(euclidean_distances(target_embedding, self.train_embeddings)).tolist()
        pairs = [(distance, index) for distance, index in zip(distances, range(len(distances)))]

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        top_pairs = list()
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, d) in top_pairs]


class EuclideanDistanceThresholdExampleSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = "sentence-transformers/all-mpnet-base-v2"
        # self.top_distances = list()
        self.threshold = 0.85

        from sentence_transformers import SentenceTransformer
        self.bert_model = SentenceTransformer(self.SELECT_MODEL, device="cpu")
        self.train_embeddings = self.bert_model.encode(self.train_questions)

    def get_examples(self, target, num_example, cross_domain=False):
        target_embedding = self.bert_model.encode([target["question"]])

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import euclidean_distances
        distances = np.squeeze(euclidean_distances(target_embedding, self.train_embeddings)).tolist()
        pairs = [(distance, index) for distance, index in zip(distances, range(len(distances)))]

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        top_pairs = list()
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if (cross_domain and similar_db_id == target["db_id"]) or d > self.threshold:
                continue
            top_pairs.append((index, d))
            # self.top_distances.append(d)
            if len(top_pairs) >= num_example:
                break
        # print("mean", np.mean(self.top_distances))    # 0.822
        # print("std", np.std(self.top_distances, ddof=1))  # 0.144
        # print("max", max(self.top_distances)) # 1.166

        return [train_json[index] for (index, d) in top_pairs]


class EuclideanDistanceSkeletonSimilarThresholdSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = "sentence-transformers/all-mpnet-base-v2"
        self.threshold = 0.85
        self.mask_token = "<mask>"  # the "<mask>" is the mask token of all-mpnet-base-v2
        self.value_token = "<unk>"  # the "<unk>" is the unknown token of all-mpnet-base-v2

        from sentence_transformers import SentenceTransformer
        train_mask_questions = mask_question_with_schema_linking(self.train_json, mask_tag=self.mask_token, value_tag=self.value_token)
        self.bert_model = SentenceTransformer(self.SELECT_MODEL, device="cpu")
        self.train_embeddings = self.bert_model.encode(train_mask_questions)

    def get_examples(self, target, num_example, cross_domain=False):
        target_mask_question = mask_question_with_schema_linking([target], mask_tag=self.mask_token, value_tag=self.value_token)
        target_embedding = self.bert_model.encode(target_mask_question)

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import euclidean_distances
        distances = np.squeeze(euclidean_distances(target_embedding, self.train_embeddings)).tolist()
        pairs = [(distance, index) for distance, index in zip(distances, range(len(distances)))]

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        top_pairs = list()
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            # Skeleton similarity
            if jaccard_similarity(train_json[index]["query_skeleton"], target["query_skeleton"]) < self.threshold:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        if len(top_pairs) < num_example:
            for d, index in pairs_sorted:
                similar_db_id = train_json[index]["db_id"]
                if cross_domain and similar_db_id == target["db_id"]:
                    continue
                # Skeleton similarity
                if jaccard_similarity(train_json[index]["query_skeleton"], target["query_skeleton"]) >= self.threshold:
                    continue
                top_pairs.append((index, d))
                if len(top_pairs) >= num_example:
                    break

        return [train_json[index] for (index, d) in top_pairs]


class EuclideanDistanceQuestionMaskSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = "sentence-transformers/all-mpnet-base-v2"
        self.mask_token = "<mask>"  # the "<mask>" is the mask token of all-mpnet-base-v2
        self.value_token = "<unk>" # the "<unk>" is the unknown token of all-mpnet-base-v2

        from sentence_transformers import SentenceTransformer
        train_mask_questions = mask_question_with_schema_linking(self.train_json, mask_tag=self.mask_token, value_tag=self.value_token)
        self.bert_model = SentenceTransformer(self.SELECT_MODEL, device="cpu")
        self.train_embeddings = self.bert_model.encode(train_mask_questions)

    def get_examples(self, target, num_example, cross_domain=False):
        target_mask_question = mask_question_with_schema_linking([target], mask_tag=self.mask_token, value_tag=self.value_token)
        target_embedding = self.bert_model.encode(target_mask_question)

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import euclidean_distances
        distances = np.squeeze(euclidean_distances(target_embedding, self.train_embeddings)).tolist()
        pairs = [(distance, index) for distance, index in zip(distances, range(len(distances)))]

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        top_pairs = list()
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, d) in top_pairs]
    
    
class EuclideanDistancePreSkeletonSimilarThresholdSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = "sentence-transformers/all-mpnet-base-v2"
        self.threshold = 0.85

        from sentence_transformers import SentenceTransformer
        self.bert_model = SentenceTransformer(self.SELECT_MODEL, device="cpu")
        self.train_embeddings = self.bert_model.encode(self.train_questions)

    def get_examples(self, target, num_example, cross_domain=False):
        target_embedding = self.bert_model.encode([target["question"]])

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import euclidean_distances
        distances = np.squeeze(euclidean_distances(target_embedding, self.train_embeddings)).tolist()
        pairs = [(distance, index) for distance, index in zip(distances, range(len(distances)))]

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        top_pairs = list()
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            # Skeleton similarity
            if jaccard_similarity(train_json[index]["pre_skeleton"], target["pre_skeleton"]) < self.threshold:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        if len(top_pairs) < num_example:
            for d, index in pairs_sorted:
                similar_db_id = train_json[index]["db_id"]
                if cross_domain and similar_db_id == target["db_id"]:
                    continue
                # Skeleton similarity
                if jaccard_similarity(train_json[index]["pre_skeleton"], target["pre_skeleton"]) >= self.threshold:
                    continue
                top_pairs.append((index, d))
                if len(top_pairs) >= num_example:
                    break

        return [train_json[index] for (index, d) in top_pairs]


class EuclideanDistancePreSkeletonSimilarPlusSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = "sentence-transformers/all-mpnet-base-v2"

        from sentence_transformers import SentenceTransformer
        self.bert_model = SentenceTransformer(self.SELECT_MODEL, device="cpu")
        self.train_embeddings = self.bert_model.encode(self.train_questions)

    def get_examples(self, target, num_example, cross_domain=False):
        target_embedding = self.bert_model.encode([target["question"]])

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import euclidean_distances
        distances = np.squeeze(euclidean_distances(target_embedding, self.train_embeddings)).tolist()
        train_json = self.train_json
        for i in range(len(train_json)):
            distances[i] -= jaccard_similarity(train_json[i]["pre_skeleton"], target["pre_skeleton"])
        pairs = [(distance, index) for distance, index in zip(distances, range(len(distances)))]
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        top_pairs = list()
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, d) in top_pairs]
    

class EuclideanDistanceQuestionMaskPreSkeletonSimilarThresholdSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = "sentence-transformers/all-mpnet-base-v2"
        self.mask_token = "<mask>"  # the "<mask>" is the mask token of all-mpnet-base-v2
        self.value_token = "<unk>"  # the "<unk>" is the unknown token of all-mpnet-base-v2
        self.threshold = 0.85

        from sentence_transformers import SentenceTransformer
        train_mask_questions = mask_question_with_schema_linking(self.train_json, mask_tag=self.mask_token, value_tag=self.value_token)
        self.bert_model = SentenceTransformer(self.SELECT_MODEL, device="cpu")
        self.train_embeddings = self.bert_model.encode(train_mask_questions)

    def get_examples(self, target, num_example, cross_domain=False):
        target_mask_question = mask_question_with_schema_linking([target], mask_tag=self.mask_token, value_tag=self.value_token)
        target_embedding = self.bert_model.encode(target_mask_question)

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import euclidean_distances
        distances = np.squeeze(euclidean_distances(target_embedding, self.train_embeddings)).tolist()
        pairs = [(distance, index) for distance, index in zip(distances, range(len(distances)))]

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        top_pairs = list()
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            # Skeleton similarity
            if jaccard_similarity(train_json[index]["pre_skeleton"], target["pre_skeleton"]) < self.threshold:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        if len(top_pairs) < num_example:
            for d, index in pairs_sorted:
                similar_db_id = train_json[index]["db_id"]
                if cross_domain and similar_db_id == target["db_id"]:
                    continue
                # Skeleton similarity
                if jaccard_similarity(train_json[index]["pre_skeleton"], target["pre_skeleton"]) >= self.threshold:
                    continue
                top_pairs.append((index, d))
                if len(top_pairs) >= num_example:
                    break

        return [train_json[index] for (index, d) in top_pairs]


class EuclideanDistanceQuestionMaskPreSkeletonSimilarThresholdShiftSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        self.SELECT_MODEL = "sentence-transformers/all-mpnet-base-v2"
        self.mask_token = "<mask>"  # the "<mask>" is the mask token of all-mpnet-base-v2
        self.value_token = "<unk>"  # the "<unk>" is the unknown token of all-mpnet-base-v2
        self.threshold = 0.85

        from sentence_transformers import SentenceTransformer
        train_mask_questions = mask_question_with_schema_linking(self.train_json, mask_tag=self.mask_token, value_tag=self.value_token)
        self.bert_model = SentenceTransformer(self.SELECT_MODEL, device="cpu")
        self.train_embeddings = self.bert_model.encode(train_mask_questions)

    def get_examples(self, target, num_example, cross_domain=False):
        target_mask_question = mask_question_with_schema_linking([target], mask_tag=self.mask_token, value_tag=self.value_token)
        target_embedding = self.bert_model.encode(target_mask_question)

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import euclidean_distances
        distances = np.squeeze(euclidean_distances(target_embedding, self.train_embeddings)).tolist()
        pairs = [(distance, index) for distance, index in zip(distances, range(len(distances)))]

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        top_pairs = list()
        for d, index in pairs_sorted:
            similar_db_id = train_json[index]["db_id"]
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            # Skeleton similarity
            if jaccard_similarity(train_json[index]["pre_skeleton"], target["pre_skeleton"]) < self.threshold:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, d) in top_pairs]

# FastKassim 实现few-shot selctor
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent / "FastKASSIM"))
import numpy as np
from FastKASSIM.fkassim.FastKassim import FastKassim

class KassimSyntaxFewShotSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

        # 移除阈值参数，无需设置
        self.fk_model = FastKassim(FastKassim.LTK)  # 使用 Label Tree Kernel

        # 解析所有训练问题的语法树
        self.train_questions = [q["question"] for q in self.train_json]
        self.train_parsed = [self.fk_model.parse_document(q) for q in self.train_questions]

    def get_examples(self, target, num_example, cross_domain=False):
        target_question = target["question"]
        target_parsed = self.fk_model.parse_document(target_question)

        similarities = []
        for i, parsed_q in enumerate(self.train_parsed):
            # 计算 target 与训练题的语法树相似度
            sim = self.fk_model.compute_similarity_preparsed(target_parsed, parsed_q)
            similarities.append((i, sim))

        # 按相似度降序排列
        similarities_sorted = sorted(similarities, key=lambda x: x[1], reverse=True)

        top_pairs = []
        for index, sim in similarities_sorted:
            similar_db_id = self.train_json[index]["db_id"]
            # 仅保留跨域/同域的过滤逻辑，移除阈值判断
            if cross_domain and similar_db_id == target["db_id"]:
                continue
            # 直接添加，不考虑相似度是否低于阈值
            top_pairs.append((index, sim))
            # 达到所需示例数量后停止
            if len(top_pairs) >= num_example:
                break

        return [self.train_json[index] for index, _ in top_pairs]
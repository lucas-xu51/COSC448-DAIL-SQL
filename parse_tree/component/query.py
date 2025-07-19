import json
from typing import List, Dict, Tuple, Optional


class SchemaElement:
    """数据库实体元素（表或字段）"""
    def __init__(self, element_id: int, name: str, element_type: str):
        self.element_id = element_id  # 唯一标识（索引）
        self.name = name  # 名称（如"singer"表、"singer.id"字段）
        self.type = element_type  # 类型："table"或"column"
        self.relation = None  # 所属表（仅字段有，指向表的SchemaElement）
        self.attributes = []  # 表的字段列表（仅表有）

    def __repr__(self):
        return f"{self.element_id}: {self.name} ({self.type})"


class SchemaGraph:
    """数据库schema图，包含实体、关联权重、最短路径"""
    KeyEdge = 0.99  # 外键-主键表关联权重
    AttEdge = 0.995  # 表-字段关联权重

    def __init__(self, db_info: Dict):
        self.schema_elements: List[SchemaElement] = []  # 所有实体（表+字段）
        self.weights: List[List[float]] = []  # 实体间关联权重矩阵
        self.shortest_distance: List[List[float]] = []  # 最短路径权重矩阵
        self.pre_element: List[List[int]] = []  # 路径前驱节点矩阵

        # 构建schema_elements（表+字段）
        self._build_schema_elements(db_info)
        # 初始化权重矩阵
        self._init_weights(db_info)
        # 计算最短路径（最强关联）
        self._compute_shortest_distance()

    def _build_schema_elements(self, db_info: Dict):
        """从db_info构建表和字段的实体"""
        # 1. 添加表实体
        tables = db_info["tables"]
        for table_idx, table_name_parts in enumerate(tables):
            table_name = " ".join(table_name_parts)  # 表名（如"singer in concert"）
            table_element = SchemaElement(
                element_id=len(self.schema_elements),
                name=table_name,
                element_type="table"
            )
            self.schema_elements.append(table_element)

        # 2. 添加字段实体（跳过index=0的"*"）
        columns = db_info["columns"]
        column_to_table = db_info["column_to_table"]
        for col_idx in range(1, len(columns)):  # columns[0]是"*"，忽略
            col_meta = columns[col_idx]
            col_name_parts = col_meta[1:]  # 字段名部分（如["singer", "id"]）
            col_name = ".".join(col_name_parts)  # 字段名（如"singer.id"）
            
            # 确定所属表（通过column_to_table映射）
            table_idx = column_to_table[str(col_idx)]
            table_element = self.schema_elements[table_idx]  # 表实体在schema_elements中的索引是table_idx

            # 创建字段实体
            col_element = SchemaElement(
                element_id=len(self.schema_elements),
                name=col_name,
                element_type="column"
            )
            col_element.relation = table_element  # 关联所属表
            self.schema_elements.append(col_element)

            # 将字段添加到表的attributes中
            table_element.attributes.append(col_element)

    def _init_weights(self, db_info: Dict):
        """初始化权重矩阵：表-字段关联、外键-主键表关联"""
        num_elements = len(self.schema_elements)
        self.weights = [[0.0 for _ in range(num_elements)] for _ in range(num_elements)]

        # 1. 表与字段的AttEdge关联（表 → 字段）
        for elem in self.schema_elements:
            if elem.type == "table":  # 表实体
                for col in elem.attributes:  # 表的字段
                    self.weights[elem.element_id][col.element_id] = self.AttEdge

        # 2. 外键与主键表的KeyEdge关联（外键字段 → 主键表）
        foreign_keys = db_info["foreign_keys"]  # {外键字段index: 主键字段index}
        column_to_table = db_info["column_to_table"]
        for fk_col_idx_str, pk_col_idx in foreign_keys.items():
            fk_col_idx = int(fk_col_idx_str)
            # 外键字段实体ID计算：表数量 + (fk_col_idx - 1)（跳过columns[0]）
            num_tables = len(db_info["tables"])
            fk_col_elem_id = num_tables + (fk_col_idx - 1)
            if fk_col_elem_id >= len(self.schema_elements):
                continue  # 无效索引
            
            # 主键字段所属的表（主键表）
            pk_table_idx = column_to_table[str(pk_col_idx)]  # 主键字段的表索引
            pk_table_elem = self.schema_elements[pk_table_idx]  # 主键表实体

            # 外键字段 → 主键表的权重设为KeyEdge
            self.weights[fk_col_elem_id][pk_table_elem.element_id] = self.KeyEdge

    def _compute_shortest_distance(self):
        """用Dijkstra算法计算最短路径（最强关联，权重乘积最大）"""
        num_elements = len(self.schema_elements)
        self.shortest_distance = [[0.0 for _ in range(num_elements)] for _ in range(num_elements)]
        self.pre_element = [[-1 for _ in range(num_elements)] for _ in range(num_elements)]

        # 初始化距离矩阵（直接关联权重）
        for i in range(num_elements):
            for j in range(num_elements):
                self.shortest_distance[i][j] = self.weights[i][j]
            self.shortest_distance[i][i] = 1.0  # 自身到自身的权重为1
            self.pre_element[i][i] = i  # 自身前驱是自己

        # 对每个节点作为源点计算最短路径
        for source in range(num_elements):
            self._dijkstra(source)

    def _dijkstra(self, source: int):
        """Dijkstra算法：计算从source到所有节点的最强关联路径"""
        num_elements = len(self.schema_elements)
        local_dist = [0.0] * num_elements  # 源点到各节点的当前最大距离
        dealt = [False] * num_elements  # 标记节点是否已处理

        # 初始化距离
        for i in range(num_elements):
            local_dist[i] = self.shortest_distance[source][i]
            self.pre_element[source][i] = source  # 初始前驱为源点

        dealt[source] = True  # 源点已处理

        # 迭代处理所有节点
        while not all(dealt):
            # 找未处理节点中距离最大的节点
            max_dist = -1.0
            max_idx = -1
            for i in range(num_elements):
                if not dealt[i] and local_dist[i] > max_dist:
                    max_dist = local_dist[i]
                    max_idx = i
            if max_idx == -1:
                break  # 所有可达节点已处理

            dealt[max_idx] = True  # 标记为已处理

            # 更新通过max_idx的路径
            for i in range(num_elements):
                if not dealt[i]:
                    new_dist = local_dist[max_idx] * self.weights[max_idx][i]
                    if new_dist > local_dist[i]:
                        local_dist[i] = new_dist
                        self.pre_element[source][i] = max_idx  # 更新前驱

        # 更新源点的最短距离
        for i in range(num_elements):
            self.shortest_distance[source][i] = local_dist[i]

    # 新增：获取与指定实体相关联的所有实体
    def get_related_elements(self, target_elem: SchemaElement) -> List[SchemaElement]:
        """
        返回所有与target_elem相关联的实体（表或字段）
        关联定义：最短路径权重 > 0 的实体（即存在有效关联）
        """
        related = []
        if target_elem.element_id >= len(self.shortest_distance):
            return related  # 无效的实体ID
        
        # 遍历所有实体，筛选出最短路径权重 > 0 的实体
        for elem in self.schema_elements:
            if elem.element_id == target_elem.element_id:
                continue  # 排除自身
            # 最短路径权重 > 0 表示存在关联
            if self.shortest_distance[target_elem.element_id][elem.element_id] > 0:
                related.append(elem)
        return related

    def print_all(self):
        """打印SchemaGraph中的所有内容"""
        print("\n===== Schema Graph 完整信息 =====")
        
        # 1. 打印所有实体（表和字段）
        print("\n1. 所有实体（表和字段）：")
        for elem in self.schema_elements:
            if elem.type == "table":
                print(f"表 {elem.element_id}: {elem.name}，包含字段：{[col.name for col in elem.attributes]}")
            else:
                print(f"字段 {elem.element_id}: {elem.name}，所属表：{elem.relation.name}")
        
        # 2. 打印权重矩阵（关键关联，过滤0值）
        print("\n2. 实体间关联权重（非0值）：")
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                weight = self.weights[i][j]
                if weight > 0:
                    src_elem = self.schema_elements[i]
                    dest_elem = self.schema_elements[j]
                    print(f"实体 {i} ({src_elem.name}) → 实体 {j} ({dest_elem.name})：权重 = {weight}")
        
        # 3. 打印最短路径（示例：选取前5个实体的关键路径）
        print("\n3. 最短路径权重（示例，非0值）：")
        sample_size = min(5, len(self.shortest_distance))  # 只打印前5个实体的路径
        for i in range(sample_size):
            for j in range(len(self.shortest_distance[i])):
                dist = self.shortest_distance[i][j]
                if dist > 0 and i != j:  # 排除自身到自身
                    src_elem = self.schema_elements[i]
                    dest_elem = self.schema_elements[j]
                    print(f"实体 {i} ({src_elem.name}) 到实体 {j} ({dest_elem.name})：最短路径权重 = {dist:.4f}")


class Query:
    """封装查询信息和schema图"""
    def __init__(self, raw_question: str, question_tokens: List[str], schema_graph: SchemaGraph):
        self.sentence = {
            "raw_question": raw_question,
            "question_tokens": question_tokens  # 分词结果
        }
        self.graph = schema_graph  # 关联的schema图
        self.parse_tree = None  # 后续语法解析的树结构（预留）
        self.mapped_elements = []  # 后续短语映射结果（预留）
        self.entities = []  # 后续实体解析结果（预留）
        self.translated_sql = None  # 最终生成的SQL（预留）


def load_queries_from_jsonl(jsonl_path: str) -> List[Query]:
    """从JSONL文件加载查询，构建Query对象"""
    queries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            schema_graph = SchemaGraph(data)
            query = Query(
                raw_question=data["raw_question"],
                question_tokens=data["question"],
                schema_graph=schema_graph
            )
            queries.append(query)
    return queries


# 示例使用：打印完整的SchemaGraph
if __name__ == "__main__":
    # 替换为你的JSONL文件路径
    jsonl_path = "C:/Users/grizz/OneDrive/Desktop/COSC448/ideas/model/DAIL-SQL/parse_tree/zfiles/test.jsonl"
    queries = load_queries_from_jsonl(jsonl_path)
    
    # 打印第一个查询的SchemaGraph完整信息
    first_query = queries[0]
    print("查询原始问题:", first_query.sentence["raw_question"])
    first_query.graph.print_all()  # 调用新增的打印函数
import json
from typing import List, Dict, Tuple, Optional


class SchemaElement:
    """Database entity element (table or column)"""
    def __init__(self, element_id: int, name: str, element_type: str):
        self.element_id = element_id  # Unique identifier (index)
        self.name = name  # Name (e.g. "singer" table, "singer.id" column)
        self.type = element_type  # Type: "table" or "column"
        self.relation = None  # Parent table (only for columns, points to table SchemaElement)
        self.attributes = []  # List of columns (only for tables)

    def __repr__(self):
        return f"{self.element_id}: {self.name} ({self.type})"


class SchemaGraph:
    """Database schema graph containing entities, relationship weights, and shortest paths"""
    KeyEdge = 0.99  # Foreign key-primary key table relationship weight
    AttEdge = 0.995  # Table-column relationship weight

    def __init__(self, db_info: Dict):
        self.schema_elements: List[SchemaElement] = []  # All entities (tables + columns)
        self.weights: List[List[float]] = []  # Relationship weight matrix between entities
        self.shortest_distance: List[List[float]] = []  # Shortest path weight matrix
        self.pre_element: List[List[int]] = []  # Path predecessor matrix

        # Build schema_elements (tables + columns)
        self._build_schema_elements(db_info)
        # Initialize weight matrix
        self._init_weights(db_info)
        # Calculate shortest paths (strongest relationships)
        self._compute_shortest_distance()

    def _build_schema_elements(self, db_info: Dict):
        """Build table and column entities from db_info"""
        # 1. Add table entities
        tables = db_info["tables"]
        for table_idx, table_name_parts in enumerate(tables):
            table_name = " ".join(table_name_parts)  # Table name (e.g. "singer in concert")
            table_element = SchemaElement(
                element_id=len(self.schema_elements),
                name=table_name,
                element_type="table"
            )
            self.schema_elements.append(table_element)

        # 2. Add column entities (skip index=0 "*")
        columns = db_info["columns"]
        column_to_table = db_info["column_to_table"]
        for col_idx in range(1, len(columns)):  # columns[0] is "*", skip
            col_meta = columns[col_idx]
            col_name_parts = col_meta[1:]  # Column name parts (e.g. ["singer", "id"])
            col_name = ".".join(col_name_parts)  # Column name (e.g. "singer.id")
            
            # Determine parent table (via column_to_table mapping)
            table_idx = column_to_table[str(col_idx)]
            table_element = self.schema_elements[table_idx]  # Table entity index in schema_elements is table_idx

            # Create column entity
            col_element = SchemaElement(
                element_id=len(self.schema_elements),
                name=col_name,
                element_type="column"
            )
            col_element.relation = table_element  # Link to parent table
            self.schema_elements.append(col_element)

            # Add column to table's attributes
            table_element.attributes.append(col_element)

    def _init_weights(self, db_info: Dict):
        """Initialize weight matrix: table-column relationships, foreign key-primary key table relationships"""
        num_elements = len(self.schema_elements)
        self.weights = [[0.0 for _ in range(num_elements)] for _ in range(num_elements)]

        # 1. Table-column AttEdge relationships (table → column)
        for elem in self.schema_elements:
            if elem.type == "table":  # Table entity
                for col in elem.attributes:  # Table's columns
                    self.weights[elem.element_id][col.element_id] = self.AttEdge

        # 2. Foreign key-primary key table KeyEdge relationships (foreign key column → primary key table)
        foreign_keys = db_info["foreign_keys"]  # {foreign key column index: primary key column index}
        column_to_table = db_info["column_to_table"]
        for fk_col_idx_str, pk_col_idx in foreign_keys.items():
            fk_col_idx = int(fk_col_idx_str)
            # Foreign key column entity ID calculation: number of tables + (fk_col_idx - 1) (skip columns[0])
            num_tables = len(db_info["tables"])
            fk_col_elem_id = num_tables + (fk_col_idx - 1)
            if fk_col_elem_id >= len(self.schema_elements):
                continue  # Invalid index
            
            # Primary key column's table (primary key table)
            pk_table_idx = column_to_table[str(pk_col_idx)]  # Primary key column's table index
            pk_table_elem = self.schema_elements[pk_table_idx]  # Primary key table entity

            # Foreign key column → primary key table weight set to KeyEdge
            self.weights[fk_col_elem_id][pk_table_elem.element_id] = self.KeyEdge

    def _compute_shortest_distance(self):
        """Calculate shortest paths using Dijkstra's algorithm (strongest relationships, maximum weight product)"""
        num_elements = len(self.schema_elements)
        self.shortest_distance = [[0.0 for _ in range(num_elements)] for _ in range(num_elements)]
        self.pre_element = [[-1 for _ in range(num_elements)] for _ in range(num_elements)]

        # Initialize distance matrix (direct relationship weights)
        for i in range(num_elements):
            for j in range(num_elements):
                self.shortest_distance[i][j] = self.weights[i][j]
            self.shortest_distance[i][i] = 1.0  # Self-to-self weight is 1
            self.pre_element[i][i] = i  # Self predecessor is self

        # Calculate shortest paths for each node as source
        for source in range(num_elements):
            self._dijkstra(source)

    def _dijkstra(self, source: int):
        """Dijkstra's algorithm: calculate strongest relationship paths from source to all nodes"""
        num_elements = len(self.schema_elements)
        local_dist = [0.0] * num_elements  # Current maximum distance from source to each node
        dealt = [False] * num_elements  # Mark whether node has been processed

        # Initialize distances
        for i in range(num_elements):
            local_dist[i] = self.shortest_distance[source][i]
            self.pre_element[source][i] = source  # Initial predecessor is source

        dealt[source] = True  # Source is processed

        # Iteratively process all nodes
        while not all(dealt):
            # Find unprocessed node with maximum distance
            max_dist = -1.0
            max_idx = -1
            for i in range(num_elements):
                if not dealt[i] and local_dist[i] > max_dist:
                    max_dist = local_dist[i]
                    max_idx = i
            if max_idx == -1:
                break  # All reachable nodes processed

            dealt[max_idx] = True  # Mark as processed

            # Update paths through max_idx
            for i in range(num_elements):
                if not dealt[i]:
                    new_dist = local_dist[max_idx] * self.weights[max_idx][i]
                    if new_dist > local_dist[i]:
                        local_dist[i] = new_dist
                        self.pre_element[source][i] = max_idx  # Update predecessor

        # Update source's shortest distances
        for i in range(num_elements):
            self.shortest_distance[source][i] = local_dist[i]

    # New: Get all entities related to specified entity
    def get_related_elements(self, target_elem: SchemaElement) -> List[SchemaElement]:
        """
        Return all entities (tables or columns) related to target_elem
        Relationship definition: entities with shortest path weight > 0 (i.e. valid relationship exists)
        """
        related = []
        if target_elem.element_id >= len(self.shortest_distance):
            return related  # Invalid entity ID
        
        # Iterate through all entities, filter those with shortest path weight > 0
        for elem in self.schema_elements:
            if elem.element_id == target_elem.element_id:
                continue  # Exclude self
            # Shortest path weight > 0 means relationship exists
            if self.shortest_distance[target_elem.element_id][elem.element_id] > 0:
                related.append(elem)
        return related

    def print_all(self):
        """Print all contents of SchemaGraph"""
        print("\n===== Schema Graph Complete Information =====")
        
        # 1. Print all entities (tables and columns)
        print("\n1. All entities (tables and columns):")
        for elem in self.schema_elements:
            if elem.type == "table":
                print(f"Table {elem.element_id}: {elem.name}, contains columns: {[col.name for col in elem.attributes]}")
            else:
                print(f"Column {elem.element_id}: {elem.name}, belongs to table: {elem.relation.name}")
        
        # 2. Print weight matrix (key relationships, filter 0 values)
        print("\n2. Entity relationship weights (non-zero values):")
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                weight = self.weights[i][j]
                if weight > 0:
                    src_elem = self.schema_elements[i]
                    dest_elem = self.schema_elements[j]
                    print(f"Entity {i} ({src_elem.name}) → Entity {j} ({dest_elem.name}): weight = {weight}")
        
        # 3. Print shortest paths (example: key paths for first 5 entities)
        print("\n3. Shortest path weights (example, non-zero values):")
        sample_size = min(5, len(self.shortest_distance))  # Only print paths for first 5 entities
        for i in range(sample_size):
            for j in range(len(self.shortest_distance[i])):
                dist = self.shortest_distance[i][j]
                if dist > 0 and i != j:  # Exclude self-to-self
                    src_elem = self.schema_elements[i]
                    dest_elem = self.schema_elements[j]
                    print(f"Entity {i} ({src_elem.name}) to Entity {j} ({dest_elem.name}): shortest path weight = {dist:.4f}")


class Query:
    """Encapsulates query information and schema graph"""
    def __init__(self, raw_question: str, question_tokens: List[str], schema_graph: SchemaGraph):
        self.sentence = {
            "raw_question": raw_question,
            "question_tokens": question_tokens  # Tokenization results
        }
        self.graph = schema_graph  # Associated schema graph
        self.parse_tree = None  # Subsequent syntax parse tree structure (reserved)
        self.mapped_elements = []  # Subsequent phrase mapping results (reserved)
        self.entities = []  # Subsequent entity parsing results (reserved)
        self.translated_sql = None  # Final generated SQL (reserved)


def load_queries_from_jsonl(jsonl_path: str) -> List[Query]:
    """Load queries from JSONL file and build Query objects"""
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


# Example usage: print complete SchemaGraph
if __name__ == "__main__":
    # Replace with your JSONL file path
    jsonl_path = "C:/Users/grizz/OneDrive/Desktop/COSC448/ideas/model/DAIL-SQL/parse_tree/zfiles/test.jsonl"
    queries = load_queries_from_jsonl(jsonl_path)
    
    # Print first query's complete SchemaGraph information
    first_query = queries[0]
    print("Query raw question:", first_query.sentence["raw_question"])
    first_query.graph.print_all()  # Call new print function
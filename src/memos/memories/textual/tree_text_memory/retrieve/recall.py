from __future__ import annotations

# concurrent.futures 提供 Future/as_completed 等并发工具。
# 本文件中主要用 concurrent.futures.as_completed 按完成顺序收集多线程结果。
import concurrent.futures

# ContextThreadPoolExecutor 是带请求上下文传播能力的线程池。
# 相比普通 ThreadPoolExecutor，它可以把 trace/user 等上下文带到子线程里，便于日志和监控串联。
from memos.context.context import ContextThreadPoolExecutor

# OllamaEmbedder 是 embedding 模型封装类型。
# Retriever 本身不直接生成 query embedding，但会保存 embedder，供未来扩展或子模块使用。
from memos.embedders.factory import OllamaEmbedder

# Neo4jGraphDB 是图数据库访问层。
# 这里所有图结构查询、向量检索、全文检索、节点加载最终都通过 graph_store 完成。
from memos.graph_dbs.neo4j import Neo4jGraphDB

# 获取模块 logger，用于记录召回路径和异常信息。
from memos.log import get_logger

# TextualMemoryItem 是文本记忆的统一业务对象。
# 图数据库返回的 dict 会被转换成这个对象，供上层 Searcher/reranker 使用。
from memos.memories.textual.item import TextualMemoryItem

# EnhancedBM25 是可选的 BM25 召回器。
# 如果初始化时传入它，retrieve 会额外开启 BM25 文本相关性召回路径。
from memos.memories.textual.tree_text_memory.retrieve.bm25_util import EnhancedBM25

# ParsedTaskGoal 是 query 被 TaskGoalParser 解析后的结构。
# 其中包含 keys、tags 等字段，供图结构召回和全文召回使用。
from memos.memories.textual.tree_text_memory.retrieve.retrieval_mid_structs import ParsedTaskGoal


# 当前模块 logger。
logger = get_logger(__name__)


# GraphMemoryRetriever 统一封装多种底层召回路径。
# 它不负责 query 解析和 rerank，只负责从图数据库中找候选 memory。
class GraphMemoryRetriever:
    """
    Unified memory retriever that combines both graph-based and vector-based retrieval logic.
    """

    # 初始化图记忆召回器。
    # graph_store 是核心依赖；embedder 和 bm25_retriever 是向量/BM25 路径相关依赖。
    def __init__(
        self,
        graph_store: Neo4jGraphDB,
        embedder: OllamaEmbedder,
        bm25_retriever: EnhancedBM25 | None = None,
        include_embedding: bool = False,
    ):
        # 保存图数据库访问对象。
        # 后续所有 get_by_metadata、search_by_embedding、search_by_fulltext、get_nodes 都通过它执行。
        self.graph_store = graph_store

        # 保存 embedding 组件。
        # 当前类主要消费外部传入的 query_embedding，但仍保留 embedder 作为召回模块依赖。
        self.embedder = embedder

        # 保存可选 BM25 检索器。
        # 传入 None 时 BM25 路径会被关闭。
        self.bm25_retriever = bm25_retriever

        # 图结构/并行处理时默认最大 worker 数。
        # 当前代码中部分地方使用固定 max_workers=3，也保留该字段供扩展使用。
        self.max_workers = 10

        # filter_weight 预留给过滤权重策略。
        # 当前文件中没有直接使用它，可能是旧逻辑或后续扩展入口。
        self.filter_weight = 0.6

        # 是否启用 BM25 召回。
        # 只要 bm25_retriever 存在，就会在 retrieve 中提交 _bm25_recall 任务。
        self.use_bm25 = bool(self.bm25_retriever)

        # 控制从 graph_store 加载节点时是否包含 embedding 字段。
        # 如果上层后续还要基于结果 embedding 去重/重排，就可以打开。
        self.include_embedding = include_embedding

    # 主召回入口。
    # 对 LongTerm/User/Tool/Skill/Preference 等记忆，它会并行执行图结构召回、向量召回、可选 BM25、可选 fulltext。
    def retrieve(
        self,
        query: str,
        parsed_goal: ParsedTaskGoal,
        top_k: int,
        memory_scope: str,
        query_embedding: list[list[float]] | None = None,
        search_filter: dict | None = None,
        search_priority: dict | None = None,
        user_name: str | None = None,
        id_filter: dict | None = None,
        use_fast_graph: bool = False,
    ) -> list[TextualMemoryItem]:
        """
        Perform hybrid memory retrieval:
        - Run graph-based lookup from dispatch plan.
        - Run vector similarity search from embedded query.
        - Merge and return combined result set.

        Args:
            query (str): Original task query.
            parsed_goal (dict): parsed_goal.
            top_k (int): Number of candidates to return.
            memory_scope (str): One of ['working', 'long_term', 'user'].
            query_embedding(list of embedding): list of embedding of query
            search_filter (dict, optional): Optional metadata filters for search results.
        Returns:
            list: Combined memory items.
        """
        # 校验 memory_scope 是否属于支持的记忆类型。
        # 这里使用服务内部的标准 memory_type 名称，而不是 docstring 中的小写别名。
        if memory_scope not in [
            "WorkingMemory",
            "LongTermMemory",
            "UserMemory",
            "ToolSchemaMemory",
            "ToolTrajectoryMemory",
            "RawFileMemory",
            "SkillMemory",
            "PreferenceMemory",
        ]:
            # 不支持的 scope 直接报错，避免向 graph_store 发送不可预期的查询。
            raise ValueError(f"Unsupported memory scope: {memory_scope}")

        # WorkingMemory 是特殊路径。
        # 它不走复杂的图结构/向量/BM25 混合召回，而是直接读取当前 scope 下所有 activated 记忆。
        if memory_scope == "WorkingMemory":
            # For working memory, retrieve all entries (no session-oriented filtering)
            # 从图数据库取出所有工作记忆项，并可选包含 embedding。
            working_memories = self.graph_store.get_all_memory_items(
                scope="WorkingMemory",
                include_embedding=self.include_embedding,
                user_name=user_name,
                filter=search_filter,
                status="activated",
            )

            # 只取前 top_k 条并转成 TextualMemoryItem。
            # 注意这里没有按分数排序，顺序由 graph_store 返回结果决定。
            return [TextualMemoryItem.from_dict(record) for record in working_memories[:top_k]]

        # 对非 WorkingMemory 的 scope，并行执行多条召回路径。
        # max_workers=3 是基础路径数量：graph/vector/BM25 或 fulltext 会按条件加入。
        with ContextThreadPoolExecutor(max_workers=3) as executor:
            # Structured graph-based retrieval
            # 图结构召回：基于 parsed_goal 中的 keys/tags 查 metadata。
            future_graph = executor.submit(
                self._graph_recall,
                parsed_goal,
                memory_scope,
                user_name,
                use_fast_graph=use_fast_graph,
            )

            # Vector similarity search
            # 向量召回：基于外部传入的 query_embedding 做 embedding 相似度搜索。
            future_vector = executor.submit(
                self._vector_recall,
                query_embedding or [],
                memory_scope,
                top_k,
                search_filter=search_filter,
                search_priority=search_priority,
                user_name=user_name,
            )

            # 如果启用了 BM25，则提交 BM25 召回路径。
            # BM25 使用 query + parsed_goal.keys 组成文本查询，再在候选节点中做文本排序。
            if self.use_bm25:
                future_bm25 = executor.submit(
                    self._bm25_recall,
                    query,
                    parsed_goal,
                    memory_scope,
                    top_k=top_k,
                    user_name=user_name,
                    search_filter=id_filter,
                )

            # fast_graph 模式下额外启用 fulltext 召回。
            # 它使用 parsed_goal.keys 作为全文检索词。
            if use_fast_graph:
                future_fulltext = executor.submit(
                    self._fulltext_recall,
                    query_words=parsed_goal.keys or [],
                    memory_scope=memory_scope,
                    top_k=top_k,
                    search_filter=search_filter,
                    search_priority=search_priority,
                    user_name=user_name,
                )

            # 等待图结构召回完成并取结果。
            graph_results = future_graph.result()

            # 等待向量召回完成并取结果。
            vector_results = future_vector.result()

            # 如果启用了 BM25，读取 BM25 结果；否则用空列表统一后续合并逻辑。
            bm25_results = future_bm25.result() if self.use_bm25 else []

            # 如果启用了 fast_graph，读取 fulltext 结果；否则用空列表。
            fulltext_results = future_fulltext.result() if use_fast_graph else []

        # Merge and deduplicate by ID
        # 将多条召回路径的结果按 item.id 合并去重。
        # 如果同一个 id 在多个路径中出现，后面的路径结果会覆盖前面的路径结果。
        combined = {
            item.id: item
            for item in graph_results + vector_results + bm25_results + fulltext_results
        }

        # 返回去重后的候选列表。
        # 注意 dict 保序语义会保留最后一次赋值位置相关顺序，但这里没有再按 score 统一排序。
        return list(combined.values())

    # 从指定 cube 中做跨 cube/外部 cube 的召回。
    # 与 retrieve 类似，但这里只使用向量召回路径，并把结果类型改成 OuterMemory。
    def retrieve_from_cube(
        self,
        top_k: int,
        memory_scope: str,
        query_embedding: list[list[float]] | None = None,
        cube_name: str = "memos_cube01",
        user_name: str | None = None,
    ) -> list[TextualMemoryItem]:
        """
        Perform hybrid memory retrieval:
        - Run graph-based lookup from dispatch plan.
        - Run vector similarity search from embedded query.
        - Merge and return combined result set.

        Args:
            top_k (int): Number of candidates to return.
            memory_scope (str): One of ['working', 'long_term', 'user'].
            query_embedding(list of embedding): list of embedding of query
            cube_name: specify cube_name

        Returns:
            list: Combined memory items.
        """
        # retrieve_from_cube 仅允许这三类 scope。
        # 它不处理 Tool/Skill/Preference/RawFile 这类特殊记忆。
        if memory_scope not in ["WorkingMemory", "LongTermMemory", "UserMemory"]:
            raise ValueError(f"Unsupported memory scope: {memory_scope}")

        # 对指定 cube_name 执行向量召回。
        # 变量名 graph_results 有点泛化，但这里实际来自 _vector_recall。
        graph_results = self._vector_recall(
            query_embedding, memory_scope, top_k, cube_name=cube_name, user_name=user_name
        )

        # 将从其他 cube 取回的结果统一标记为 OuterMemory。
        # 这样上层可以把它们与当前用户本地 memory 区分开。
        for result_i in graph_results:
            result_i.metadata.memory_type = "OuterMemory"

        # Merge and deduplicate by ID
        # 只按 id 去重，不做额外排序。
        combined = {item.id: item for item in graph_results}

        # 返回外部 cube 召回结果。
        return list(combined.values())

    # plugin/simple 搜索使用的简化混合召回入口。
    # 它只走向量召回，并按 id 去重。
    def retrieve_from_mixed(
        self,
        top_k: int,
        memory_scope: str | None = None,
        query_embedding: list[list[float]] | None = None,
        search_filter: dict | None = None,
        user_name: str | None = None,
    ) -> list[TextualMemoryItem]:
        """Retrieve from mixed and memory"""
        # 使用 query_embedding 在指定 scope 或全局 scope 中做向量召回。
        vector_results = self._vector_recall(
            query_embedding or [],
            memory_scope,
            top_k,
            search_filter=search_filter,
            user_name=user_name,
        )  # Merge and deduplicate by ID

        # 按 id 去重。
        combined = {item.id: item for item in vector_results}

        # 返回召回项。
        return list(combined.values())

    # 图结构召回路径。
    # 主要根据 parsed_goal.keys 和 parsed_goal.tags 去 metadata 中找候选节点。
    def _graph_recall(
        self, parsed_goal: ParsedTaskGoal, memory_scope: str, user_name: str | None = None, **kwargs
    ) -> list[TextualMemoryItem]:
        """
        Perform structured node-based retrieval from Neo4j.
        - keys must match exactly (n.key IN keys)
        - tags must overlap with at least 2 input tags
        - scope filters by memory_type if provided
        """
        # 读取是否启用 fast_graph。
        # fast_graph 会在 metadata 查询时附加 status="activated"，并用线程池并行 post-filter 节点。
        use_fast_graph = kwargs.get("use_fast_graph", False)

        # fast_graph 分支中用于并行处理单个节点的内部函数。
        def process_node(node):
            # 取节点 metadata。
            meta = node.get("metadata", {})

            # key 是结构化召回的强匹配字段。
            node_key = meta.get("key")

            # tags 是结构化召回的弱匹配字段，要求至少有 2 个重叠。
            node_tags = meta.get("tags", []) or []

            # keep 表示该节点是否通过 keys/tags 过滤。
            keep = False

            # key equals to node_key
            # 如果 parsed_goal.keys 中包含节点 key，则直接保留。
            if parsed_goal.keys and node_key in parsed_goal.keys:
                keep = True

            # overlap tags more than 2
            # 如果 key 没命中，则用 tags 重叠数量做召回条件。
            elif parsed_goal.tags:
                # 节点 tag 转小写，减少大小写差异影响。
                node_tags_list = [tag.lower() for tag in node_tags]

                # 计算节点 tags 与 parsed_goal.tags 的交集数量。
                overlap = len(set(node_tags_list) & set(parsed_goal.tags))

                # 至少两个 tag 重叠才保留。
                if overlap >= 2:
                    keep = True

            # 通过过滤时将 dict 转成 TextualMemoryItem。
            if keep:
                return TextualMemoryItem.from_dict(node)

            # 未通过过滤时返回 None，后续会过滤掉。
            return None

        # 非 fast_graph 分支：顺序加载并过滤候选节点。
        if not use_fast_graph:
            # candidate_ids 收集 metadata 查询命中的节点 id。
            candidate_ids = set()

            # 1) key-based OR branch
            # 如果解析目标包含 keys，则按 key in keys + memory_type 查询候选 id。
            if parsed_goal.keys:
                # key_filters 表示“key 命中且 memory_type 匹配”。
                key_filters = [
                    {"field": "key", "op": "in", "value": parsed_goal.keys},
                    {"field": "memory_type", "op": "=", "value": memory_scope},
                ]

                # 从 graph_store 按 metadata 查询节点 id。
                key_ids = self.graph_store.get_by_metadata(key_filters, user_name=user_name)

                # 合并进候选集合。
                candidate_ids.update(key_ids)

            # 2) tag-based OR branch
            # 如果解析目标包含 tags，则按 tags contains + memory_type 查询候选 id。
            if parsed_goal.tags:
                # tag_filters 表示“tags 包含目标 tags 且 memory_type 匹配”。
                tag_filters = [
                    {"field": "tags", "op": "contains", "value": parsed_goal.tags},
                    {"field": "memory_type", "op": "=", "value": memory_scope},
                ]

                # 从 graph_store 按 tag metadata 查询节点 id。
                tag_ids = self.graph_store.get_by_metadata(tag_filters, user_name=user_name)

                # key 和 tag 两条分支是 OR 关系，所以统一 update。
                candidate_ids.update(tag_ids)

            # No matches → return empty
            # 没有任何 key/tag 候选时，图结构路径直接返回空。
            if not candidate_ids:
                return []

            # Load nodes and post-filter
            # 批量加载候选节点详情。
            node_dicts = self.graph_store.get_nodes(
                list(candidate_ids), include_embedding=self.include_embedding, user_name=user_name
            )

            # final_nodes 保存通过二次过滤的节点。
            final_nodes = []

            # 对每个候选节点再次做精确判断。
            # 这一步是为了处理 get_by_metadata 的 contains/in 查询可能较宽松的问题。
            for node in node_dicts:
                # 取节点 metadata。
                meta = node.get("metadata", {})

                # 取节点 key。
                node_key = meta.get("key")

                # 取节点 tags。
                node_tags = meta.get("tags", []) or []

                # 默认不保留。
                keep = False

                # key equals to node_key
                # key 精确命中则保留。
                if parsed_goal.keys and node_key in parsed_goal.keys:
                    keep = True

                # overlap tags more than 2
                # 否则判断 tags 与目标 tags 是否至少有两个重叠。
                elif parsed_goal.tags:
                    overlap = len(set(node_tags) & set(parsed_goal.tags))
                    if overlap >= 2:
                        keep = True

                # 通过过滤后转成 TextualMemoryItem 并加入结果。
                if keep:
                    final_nodes.append(TextualMemoryItem.from_dict(node))

            # 返回图结构召回结果。
            return final_nodes

        # fast_graph 分支：metadata 查询加 activated 状态，并使用线程池并行 post-filter。
        else:
            # candidate_ids 收集候选节点 id。
            candidate_ids = set()

            # 1) key-based OR branch
            # key 分支：按 key + memory_type + activated 查询。
            if parsed_goal.keys:
                # 构造 key 查询过滤条件。
                key_filters = [
                    {"field": "key", "op": "in", "value": parsed_goal.keys},
                    {"field": "memory_type", "op": "=", "value": memory_scope},
                ]

                # fast_graph 下显式要求 status="activated"。
                key_ids = self.graph_store.get_by_metadata(
                    key_filters, user_name=user_name, status="activated"
                )

                # 合并 key 候选。
                candidate_ids.update(key_ids)

            # 2) tag-based OR branch
            # tag 分支：按 tags + memory_type + activated 查询。
            if parsed_goal.tags:
                # 构造 tag 查询过滤条件。
                tag_filters = [
                    {"field": "tags", "op": "contains", "value": parsed_goal.tags},
                    {"field": "memory_type", "op": "=", "value": memory_scope},
                ]

                # fast_graph 下也显式要求 activated。
                tag_ids = self.graph_store.get_by_metadata(
                    tag_filters, user_name=user_name, status="activated"
                )

                # 合并 tag 候选。
                candidate_ids.update(tag_ids)

            # No matches → return empty
            # 没有候选 id 时直接返回空。
            if not candidate_ids:
                return []

            # Load nodes and post-filter
            # 批量加载候选节点。
            node_dicts = self.graph_store.get_nodes(
                list(candidate_ids), include_embedding=self.include_embedding, user_name=user_name
            )

            # final_nodes 先声明，后面由并行过滤结果生成。
            final_nodes = []

            # 使用线程池并行处理每个节点。
            # 这在候选节点较多、post-filter 有额外逻辑时可以减少延迟。
            with ContextThreadPoolExecutor(max_workers=3) as executor:
                # 提交每个节点的 process_node 任务。
                # futures 的 value 是原始下标，用来恢复结果顺序。
                futures = {
                    executor.submit(process_node, node): i for i, node in enumerate(node_dicts)
                }

                # 用固定长度列表暂存结果，保证最终顺序与 node_dicts 一致。
                temp_results = [None] * len(node_dicts)

                # 按完成顺序收集结果。
                for future in concurrent.futures.as_completed(futures):
                    # 找回该 future 对应的原始下标。
                    original_index = futures[future]

                    # 获取 process_node 返回的 TextualMemoryItem 或 None。
                    result = future.result()

                    # 写回原始下标位置。
                    temp_results[original_index] = result

                # 过滤掉未命中的 None。
                final_nodes = [result for result in temp_results if result is not None]

            # 返回 fast_graph 图结构召回结果。
            return final_nodes

    # 向量召回路径。
    # 它用一个或多个 query embedding 到 graph_store.search_by_embedding 中检索候选节点。
    def _vector_recall(
        self,
        query_embedding: list[list[float]],
        memory_scope: str,
        top_k: int = 20,
        max_num: int = 20,
        status: str = "activated",
        cube_name: str | None = None,
        search_filter: dict | None = None,
        search_priority: dict | None = None,
        user_name: str | None = None,
    ) -> list[TextualMemoryItem]:
        """
        Perform vector-based similarity retrieval using query embedding.
        # TODO: tackle with post-filter and pre-filter(5.18+) better.
        """
        # 没有 query_embedding 时无法进行向量检索。
        if not query_embedding:
            return []

        # 对单个向量执行一次 graph_store.search_by_embedding。
        # search_priority 会作为 search_filter 参数传入，search_filter 会作为 filter 参数传入。
        def search_single(vec, search_priority=None, search_filter=None):
            return (
                self.graph_store.search_by_embedding(
                    vector=vec,
                    top_k=top_k,
                    status=status,
                    scope=memory_scope,
                    cube_name=cube_name,
                    search_filter=search_priority,
                    filter=search_filter,
                    user_name=user_name,
                )
                or []
            )

        # Path A：普通向量召回，不使用 search_priority。
        def search_path_a():
            """Path A: search without priority"""
            # 收集所有 query embedding 的命中结果。
            path_a_hits = []

            # 为每个 query embedding 并发执行一次检索。
            with ContextThreadPoolExecutor() as executor:
                # 最多使用前 max_num 个 query embedding，避免 query 扩展过多导致请求爆炸。
                futures = [
                    executor.submit(search_single, vec, None, search_filter)
                    for vec in query_embedding[:max_num]
                ]

                # 按完成顺序合并结果。
                for f in concurrent.futures.as_completed(futures):
                    path_a_hits.extend(f.result() or [])

            # 返回普通向量召回命中。
            return path_a_hits

        # Path B：带 search_priority 的向量召回。
        # 这相当于给高优先级 metadata 条件另开一条召回分支。
        def search_path_b():
            """Path B: search with priority"""
            # 没有 search_priority 时该路径关闭。
            if not search_priority:
                return []

            # 收集优先级召回命中。
            path_b_hits = []

            # 同样对多个 query embedding 并发检索。
            with ContextThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(search_single, vec, search_priority, search_filter)
                    for vec in query_embedding[:max_num]
                ]

                # 合并每个 future 的结果。
                for f in concurrent.futures.as_completed(futures):
                    path_b_hits.extend(f.result() or [])

            # 返回优先级向量召回命中。
            return path_b_hits

        # Execute both paths concurrently
        # 并发执行普通向量路径和优先级向量路径。
        all_hits = []
        with ContextThreadPoolExecutor(max_workers=2) as executor:
            # 提交 Path A。
            path_a_future = executor.submit(search_path_a)

            # 提交 Path B。
            path_b_future = executor.submit(search_path_b)

            # 合并 Path A 结果。
            all_hits.extend(path_a_future.result())

            # 合并 Path B 结果。
            all_hits.extend(path_b_future.result())

        # 如果两个路径都没有命中，直接返回空。
        if not all_hits:
            return []

        # merge and deduplicate, keeping highest score per ID
        # 将所有 hit 按 id 去重，并保留每个 id 的最高 score。
        id_to_score = {}

        # 遍历 graph_store.search_by_embedding 返回的原始 hit。
        for r in all_hits:
            # 取 hit 的 id。
            rid = r.get("id")

            # 有 id 才能进一步加载节点。
            if rid:
                # 统一转成字符串，并去掉外层引号。
                rid = str(rid).strip("\"'")

                # 读取相似度分数，缺省为 0。
                score = r.get("score", 0.0)

                # 如果该 id 第一次出现，或当前 score 更高，则更新最高分。
                if rid not in id_to_score or score > id_to_score[rid]:
                    id_to_score[rid] = score

        # Sort IDs by score (descending) to preserve ranking
        # 按最高 score 降序排列节点 id。
        sorted_ids = sorted(id_to_score.keys(), key=lambda x: id_to_score[x], reverse=True)

        # 根据排序后的 ids 批量加载完整节点内容。
        node_dicts = (
            self.graph_store.get_nodes(
                sorted_ids,
                include_embedding=self.include_embedding,
                cube_name=cube_name,
                user_name=user_name,
            )
            or []
        )

        # Restore score-based order and inject scores into metadata
        # 建立 node_id -> node dict 映射，便于按 sorted_ids 恢复顺序。
        id_to_node = {}

        # 遍历加载出的节点。
        for n in node_dicts:
            # 取节点 id。
            node_id = n.get("id")

            # 有 id 才能参与映射。
            if node_id:
                # Ensure ID is a string and strip any surrounding quotes
                # 同样标准化 id 字符串。
                node_id = str(node_id).strip("\"'")

                # 记录到映射中。
                id_to_node[node_id] = n

        # ordered_nodes 保存最终按 score 排序的节点 dict。
        ordered_nodes = []

        # 按 sorted_ids 顺序恢复节点。
        for rid in sorted_ids:
            # Ensure rid is normalized for matching
            # 标准化 rid，确保能和 id_to_node 对上。
            rid_normalized = str(rid).strip("\"'")

            # 只保留成功加载出完整节点的 id。
            if rid_normalized in id_to_node:
                # 取完整节点。
                node = id_to_node[rid_normalized]

                # Inject similarity score as relativity
                # 确保 metadata 字段存在。
                if "metadata" not in node:
                    node["metadata"] = {}

                # 把向量相似度 score 写入 metadata.relativity。
                # 上层 rerank/filter 可以用这个字段表示初始相关度。
                node["metadata"]["relativity"] = id_to_score.get(rid, 0.0)

                # 加入最终有序节点列表。
                ordered_nodes.append(node)

        # 将节点 dict 转成 TextualMemoryItem 返回。
        return [TextualMemoryItem.from_dict(n) for n in ordered_nodes]

    # BM25 召回路径。
    # 它先根据 memory_type 和 id_filter 缩小候选节点，再用 BM25 对候选文本排序。
    def _bm25_recall(
        self,
        query: str,
        parsed_goal: ParsedTaskGoal,
        memory_scope: str,
        top_k: int = 20,
        user_name: str | None = None,
        search_filter: dict | None = None,
    ) -> list[TextualMemoryItem]:
        """
        Perform BM25-based retrieval.
        """
        # 没有 BM25 retriever 时直接关闭该路径。
        if not self.bm25_retriever:
            return []

        # BM25 候选池至少要限制 memory_type。
        key_filters = [
            {"field": "memory_type", "op": "=", "value": memory_scope},
        ]

        # corpus_name is user_name + user_id
        # corpus_name 用于 BM25 索引/缓存命名，默认以 user_name 区分用户语料。
        corpus_name = f"{user_name}" if user_name else ""

        # 如果传入 search_filter，则把它们转换成 metadata 等值过滤条件。
        if search_filter is not None:
            # 遍历 filter 中每个字段。
            for key in search_filter:
                # 取字段值。
                value = search_filter[key]

                # 加入等值过滤条件。
                key_filters.append({"field": key, "op": "=", "value": value})

            # 将过滤值拼到 corpus_name，避免不同过滤条件复用同一个 BM25 语料缓存。
            corpus_name += "".join(list(search_filter.values()))

        # 根据 metadata 过滤条件获取候选节点 id。
        candidate_ids = self.graph_store.get_by_metadata(
            key_filters, user_name=user_name, status="activated"
        )

        # 加载候选节点详情。
        node_dicts = self.graph_store.get_nodes(
            list(candidate_ids), include_embedding=self.include_embedding, user_name=user_name
        )

        # BM25 查询文本由原始 query 和 parsed_goal.keys 合并去重组成。
        # 这样可同时利用自然语言 query 和结构化关键词。
        bm25_query = " ".join(list({query, *parsed_goal.keys}))

        # 在候选节点集合中执行 BM25 搜索。
        bm25_results = self.bm25_retriever.search(
            bm25_query, node_dicts, top_k=top_k, corpus_name=corpus_name
        )

        # 将 BM25 结果转成 TextualMemoryItem。
        return [TextualMemoryItem.from_dict(n) for n in bm25_results]

    # 全文检索路径。
    # 它直接调用 graph_store.search_by_fulltext，用 query_words 在数据库全文索引中找候选。
    def _fulltext_recall(
        self,
        query_words: list[str],
        memory_scope: str,
        top_k: int = 20,
        max_num: int = 5,
        status: str = "activated",
        cube_name: str | None = None,
        search_filter: dict | None = None,
        search_priority: dict | None = None,
        user_name: str | None = None,
    ):
        """Perform fulltext-based retrieval.
        Args:
            query_words: list of query words
            memory_scope: memory scope
            top_k: top k results
            max_num: max number of query words
            status: status
            cube_name: cube name
            search_filter: search filter
            search_priority: search priority
            user_name: user name
        Returns:
            list of TextualMemoryItem
        """
        # 没有关键词时无法做全文检索。
        if not query_words:
            return []

        # 记录用于全文检索的关键词。
        logger.info(f"[FULLTEXT] query_words: {query_words}")

        # 调用图数据库全文索引检索。
        # search_priority 传给 search_filter，普通 search_filter 传给 filter，与向量召回路径保持一致。
        all_hits = self.graph_store.search_by_fulltext(
            query_words=query_words,
            top_k=top_k,
            status=status,
            scope=memory_scope,
            cube_name=cube_name,
            search_filter=search_priority,
            filter=search_filter,
            user_name=user_name,
        )

        # 没有命中时直接返回空。
        if not all_hits:
            return []

        # merge and deduplicate, keeping highest score per ID
        # 全文检索可能返回重复 id，这里按 id 保留最高 score。
        id_to_score = {}

        # 遍历全文检索 hit。
        for r in all_hits:
            # 取 hit id。
            rid = r.get("id")

            # 有 id 才能加载完整节点。
            if rid:
                # Ensure ID is a string and strip any surrounding quotes
                # 标准化 id。
                rid = str(rid).strip("\"'")

                # 读取全文检索分数。
                score = r.get("score", 0.0)

                # 保留该 id 的最高分。
                if rid not in id_to_score or score > id_to_score[rid]:
                    id_to_score[rid] = score

        # Sort IDs by score (descending) to preserve ranking
        # 按全文检索 score 降序排列 id。
        sorted_ids = sorted(id_to_score.keys(), key=lambda x: id_to_score[x], reverse=True)

        # 批量加载完整节点。
        node_dicts = (
            self.graph_store.get_nodes(
                sorted_ids,
                include_embedding=self.include_embedding,
                cube_name=cube_name,
                user_name=user_name,
            )
            or []
        )

        # Restore score-based order and inject scores into metadata
        # 建立 id 到节点 dict 的映射。
        id_to_node = {}

        # 遍历加载出来的节点。
        for n in node_dicts:
            # 取节点 id。
            node_id = n.get("id")

            # 有 id 才能映射。
            if node_id:
                # Ensure ID is a string and strip any surrounding quotes
                # 标准化节点 id。
                node_id = str(node_id).strip("\"'")

                # 记录节点。
                id_to_node[node_id] = n

        # ordered_nodes 保存最终按全文 score 排序的节点。
        ordered_nodes = []

        # 按 sorted_ids 恢复顺序。
        for rid in sorted_ids:
            # Ensure rid is normalized for matching
            # 标准化 rid。
            rid_normalized = str(rid).strip("\"'")

            # 只保留成功加载出来的节点。
            if rid_normalized in id_to_node:
                # 取节点。
                node = id_to_node[rid_normalized]

                # Inject similarity score as relativity
                # 确保 metadata 存在。
                if "metadata" not in node:
                    node["metadata"] = {}

                # 将全文检索 score 注入 metadata.relativity。
                node["metadata"]["relativity"] = id_to_score.get(rid, 0.0)

                # 加入最终节点列表。
                ordered_nodes.append(node)

        # 转成 TextualMemoryItem 返回。
        return [TextualMemoryItem.from_dict(n) for n in ordered_nodes]
